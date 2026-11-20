"""Compile the embedded CUDA extension locally without importing SGLang."""

import ast
import base64
import hashlib
import sys
import types
import zlib
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
source_path = Path(arguments[0]) if arguments else Path("submission.py")
source_text = source_path.read_text()
tree = ast.parse(source_text)
assignments = {
    target.id: node.value
    for node in tree.body
    if isinstance(node, ast.Assign)
    for target in node.targets
    if isinstance(target, ast.Name)
}
binding_functions = None
cpp_source = ""
if "--pm" in sys.argv:
    namespace = {"__name__": "_local_submission_compile_"}
    exec(compile(source_text, str(source_path), "exec"), namespace)
    source = namespace["_PM_CUDA"]
    cpp_source = namespace["_PM_CPP"]
    binding_functions = ["pm_launch"]
elif "--o256" in sys.argv:
    # v56's isolated M256 O-projection specialization.
    namespace = {"__name__": "_local_submission_compile_"}
    exec(compile(source_text, str(source_path), "exec"), namespace)
    source = namespace["_o256_rl_source"](namespace["_o256_CONFIGS"])
    source = source.replace(
        'cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q)',
        'cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",(void**)&encode,cudaEnableDefault,&q)',
    )
    cpp_source = namespace["_o256_RL_CPP"]
    binding_functions = ["rl_launch"]
elif "--q32" in sys.argv:
    namespace = {"__name__": "_local_submission_compile_"}
    exec(compile(source_text, str(source_path), "exec"), namespace)
    source = namespace["_q32_cuda_source"]()
    cpp_source = namespace["_q32_CPP"]
    binding_functions = ["m32x_q_launch" if "m32x_q_launch" in cpp_source else "m32_launch"]
elif "--pwide-v145" in sys.argv:
    # Isolate v145's half-width BF16 DSMEM wide-Pooler kernels.  The submitted
    # module uses no-implicit-header bindings; this local path feeds the same
    # generated CUDA into the older workstation PyTorch build.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    source = namespace["_v145_cuda_source"](
        list(namespace["_V145_PW_CONFIGS"])
    )
    cpp_source = (
        "#include <stdint.h>\n"
        "int qb_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);"
    )
    binding_functions = ["qb_launch"]
elif "--pwide" in sys.argv:
    # v52's persistent Pooler source is generated from the qbig descriptor
    # kernel plus the optional two-consumer-warpgroup transformation.
    namespace = {"__name__": "_local_submission_compile_"}
    exec(compile(source_text, str(source_path), "exec"), namespace)
    cpp_source, source = namespace["_pw_cuda_source"]()
    binding_functions = ["qb_launch"]
elif "--o16" in sys.argv:
    namespace = {"__name__": "_local_submission_compile_"}
    exec(compile(source_text, str(source_path), "exec"), namespace)
    cpp_source, source, symbol = namespace["mt_M16TRACE_SOURCES"]["o"]
    binding_functions = [symbol]
elif "--r32" in sys.argv:
    namespace = {"__name__": "_local_submission_compile_"}
    exec(compile(source_text, str(source_path), "exec"), namespace)
    source = namespace["_r32_cuda_source"]()
    cpp_source = namespace["_r32_CPP"]
    binding_functions = ["rr_launch"]
elif "--sk2-prodcorr-v119" in sys.argv:
    # Isolate v119's producer-warp C12 correction kernels for exact SM90a
    # ptxas resource and spill validation.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    common = namespace["CUDA_SRC"].replace(
        "@@MBAR@@", namespace["MBAR_HELPERS"]
    ).replace("@@BODY@@", namespace["WG_HELPERS"])
    source = common + namespace["MAP_SRC"] + namespace["_v119_sk2p_src"](True)
    cpp_source = (
        "void sk2p(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
        "int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,"
        "torch::Tensor);\n#include <vector>\n#include <pybind11/stl.h>\n"
        "std::vector<int64_t> sk2p_info(int64_t,int64_t,int64_t);"
    )
    binding_functions = ["sk2p", "sk2p_info"]
elif "--cgemv-radix-v143" in sys.argv:
    # Isolate v143's lossless 32-bit escape-leaf GEMV kernels.  This keeps the
    # ptxas/resource check independent of the large production translation
    # unit while compiling the exact transformed kernel source.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    common = namespace["CUDA_SRC"].replace(
        "@@MBAR@@", namespace["MBAR_HELPERS"]
    ).replace("@@BODY@@", "")
    source = (common + namespace["RGEMV_SRC"] +
              namespace["_v143_radix_kernel_source"]() +
              namespace["_V143_WRAPPER_SRC"])
    cpp_source = (
        "void cgemv_radix(torch::Tensor,torch::Tensor,torch::Tensor,"
        "torch::Tensor,int64_t,torch::Tensor,int64_t,int64_t,int64_t);\n"
        "int64_t cgemv_radix_info(int64_t);"
    )
    binding_functions = ["cgemv_radix", "cgemv_radix_info"]
elif "--sk2-rowcorr-v118" in sys.argv:
    # Isolate v118's seven row-owned C12 correction kernels for exact SM90a
    # ptxas resource and spill validation.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    common = namespace["CUDA_SRC"].replace(
        "@@MBAR@@", namespace["MBAR_HELPERS"]
    ).replace("@@BODY@@", namespace["WG_HELPERS"])
    source = common + namespace["MAP_SRC"] + namespace["_v118_sk2r_src"](True)
    cpp_source = (
        "void sk2r(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
        "int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,"
        "torch::Tensor);\n#include <vector>\n#include <pybind11/stl.h>\n"
        "std::vector<int64_t> sk2r_info(int64_t,int64_t,int64_t);"
    )
    binding_functions = ["sk2r", "sk2r_info"]
elif "--qkv-direct-v117" in sys.argv:
    # Isolate v117's three direct-register QKV kernels for exact SM90a ptxas
    # resource and spill validation.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    source = namespace["_V117_QKV_CUDA"]
    cpp_source = namespace["_V117_QKV_CPP"]
    binding_functions = ["qkv_direct", "qkv_direct_info"]
elif "--sk2-cl2-v116" in sys.argv:
    # Isolate the six v116 small-QKV cluster split-K kernels.  Compiling the
    # exact generated SM90a translation unit locally catches template,
    # register-spill, and dynamic-shared-memory regressions before H200 time.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    common = namespace["CUDA_SRC"].replace(
        "@@MBAR@@", namespace["MBAR_HELPERS"]
    ).replace("@@BODY@@", namespace["WG_HELPERS"])
    source = common + namespace["MAP_SRC"] + namespace["_v116_sk2_src"](True)
    cpp_source = (
        "void sk2(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
        "int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,"
        "torch::Tensor);\nint64_t sk2_cand_info(int64_t,int64_t,int64_t);"
    )
    binding_functions = ["sk2", "sk2_cand_info"]
elif "--sk2-localbase-v114" in sys.argv:
    # Isolate the seven v114 per-CTA-C12-base SK2 twins. The production
    # kernels are omitted so ptxas resource/SASS inspection stays bounded.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    common = namespace["CUDA_SRC"].replace(
        "@@MBAR@@", namespace["MBAR_HELPERS"]
    ).replace("@@BODY@@", namespace["WG_HELPERS"])
    source = (common + namespace["MAP_SRC"] +
              namespace["_v114_sk2_localbase_src"](True))
    cpp_source = (
        "void sk2_lb(torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, "
        "int64_t, int64_t, int64_t, int64_t, torch::Tensor);\n"
        "int64_t sk2_lb_info(int64_t, int64_t);"
    )
    binding_functions = ["sk2_lb", "sk2_lb_info"]
elif "--sk2-l2-v113" in sys.argv:
    # Isolate the seven v113 SK2 L2-policy twins. The production kernels are
    # omitted so ptxas resource/SASS inspection stays bounded.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    common = namespace["CUDA_SRC"].replace(
        "@@MBAR@@", namespace["MBAR_HELPERS"]
    ).replace("@@BODY@@", namespace["WG_HELPERS"])
    source = (common + namespace["MAP_SRC"] +
              namespace["_v113_sk2_l2_src"](True))
    cpp_source = (
        "void sk2_l2(torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, "
        "int64_t, int64_t, int64_t, torch::Tensor);\n"
        "int64_t sk2_l2_info(int64_t, int64_t);"
    )
    binding_functions = ["sk2_l2", "sk2_l2_info"]
elif "--sk2-bf16-v112" in sys.argv:
    # Isolate the seven v112 BF16 shared-partial SK2 twins. The production
    # kernels are omitted so ptxas resource/SASS inspection stays bounded.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    common = namespace["CUDA_SRC"].replace(
        "@@MBAR@@", namespace["MBAR_HELPERS"]
    ).replace("@@BODY@@", namespace["WG_HELPERS"])
    source = (common + namespace["MAP_SRC"] +
              namespace["_v112_sk2_bf16_src"](True))
    cpp_source = (
        "void sk2_bf16(torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, "
        "int64_t, int64_t, int64_t, torch::Tensor);\n"
        "int64_t sk2_bf16_info(int64_t, int64_t);"
    )
    binding_functions = ["sk2_bf16", "sk2_bf16_info"]
elif "--router-c12-v111" in sys.argv:
    # Isolate the two v111 dual-C12 router candidates.  The production
    # extension is left out so ptxas resource/SASS inspection stays bounded.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    common = namespace["CUDA_SRC"].replace(
        "@@MBAR@@", namespace["MBAR_HELPERS"]
    ).replace("@@BODY@@", namespace["WG_HELPERS"])
    source = (common + namespace["wgmma_defs"]([64, 128])
              + namespace["MAP_SRC"] + namespace["_V111_SK2_SRC"]
              + namespace["_V111_WIDE_SRC"])
    cpp_source = (
        "void sk2c12(torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, "
        "int64_t, int64_t, int64_t, torch::Tensor);\n"
        "void widec12(torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, "
        "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);"
    )
    binding_functions = ["sk2c12", "widec12"]
elif "--wgmma-v109" in sys.argv:
    # Compile the two v109 striped-accumulator candidates in one SM90a
    # translation unit.  prepare()/kernel_fn are not called.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    source = (namespace["_v109_qkv_source"]() +
              "\n" + namespace["_v109_router_source"]())
    source = source.replace(
        'cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q)',
        'cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",(void**)&encode,cudaEnableDefault,&q)',
    )
    cpp_source = (
        "#include <torch/extension.h>\n#include <stdint.h>\n"
        "int qb_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);\n"
        "void rk4(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);\n"
        "int64_t rk4_info();"
    )
    binding_functions = ["qb_launch", "rk4", "rk4_info"]
elif "--gqm-v108" in sys.argv:
    # Isolate the v108 four-CTA activation-multicast WGMMA route.  Provide the
    # same harmless SGLang stub used by the full-submission compile paths.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    common = namespace["CUDA_SRC"].replace(
        "@@MBAR@@", namespace["MBAR_HELPERS"]
    ).replace("@@BODY@@", namespace["WG_HELPERS"])
    source = (common + namespace["wgmma_ss_defs"]([128, 256])
              + namespace["MAP_SRC"] + namespace["_V108_GQM_CUDA"])
    cpp_source = (
        "void gqm(torch::Tensor, torch::Tensor, torch::Tensor, "
        "int64_t, int64_t, int64_t, int64_t);\n"
        "int64_t gqm_info(int64_t);"
    )
    binding_functions = ["gqm", "gqm_info"]
elif "--sk13-v107" in sys.argv:
    # Isolate the incumbent SK2 and v107 SK13 kernels.  The submission's
    # embedded alternate module imports one SGLang symbol at definition time;
    # provide a harmless stub because prepare()/kernel_fn are never called by
    # this compile-only resource/SASS path.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    common = namespace["CUDA_SRC"].replace(
        "@@MBAR@@", namespace["MBAR_HELPERS"]
    ).replace("@@BODY@@", namespace["WG_HELPERS"])
    source = (common + namespace["MAP_SRC"] + namespace["_sk2_src"](True)
              + namespace["_v107_sk13_src"](True))
    cpp_source = namespace["CPP_SRC"]
    binding_functions = ["sk2", "sk13", "sk13_info"]
elif ("--cgemv-v104" in sys.argv or "--cgemv-v105" in sys.argv
      or "--cgemv-v106" in sys.argv):
    # Isolate the generic C12 GEMV family.  Evaluating the complete
    # submission would import its embedded alternate module (and SGLang), which
    # is unnecessary for ptxas resource/SASS inspection.
    def first_literal_assignment(name):
        for item in tree.body:
            if (isinstance(item, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name
                            for target in item.targets)
                    and isinstance(item.value, ast.Constant)):
                return ast.literal_eval(item.value)
        raise KeyError(name)

    full_cuda = first_literal_assignment("RGEMV_SRC")
    value_by_name = {}
    version = ("v106" if "--cgemv-v106" in sys.argv else
               "v105" if "--cgemv-v105" in sys.argv else "v104")
    # v106 is an append-only experiment over v105 and still relies on the
    # v105 source rewrite that injects cgemv_info().  Its own appended CUDA
    # block has no launch anchor, so use the incumbent marker here.
    launch_marker = "_v105_launch_anchor" if version == "v106" else f"_{version}_launch_anchor"
    variant_start = None
    for item in tree.body:
        if not isinstance(item, ast.Assign):
            continue
        names = [target.id for target in item.targets if isinstance(target, ast.Name)]
        if launch_marker in names:
            variant_start = item.lineno
        if variant_start is not None:
            for name in names:
                try:
                    value_by_name[name] = ast.literal_eval(item.value)
                except (ValueError, TypeError):
                    pass
        if (variant_start is not None and "RGEMV_SRC" in names
                and isinstance(item.value, ast.Call)
                and isinstance(item.value.func, ast.Attribute)
                and item.value.func.attr == "replace"):
            old_node, new_node = item.value.args[:2]
            old = (value_by_name[old_node.id] if isinstance(old_node, ast.Name)
                   else ast.literal_eval(old_node))
            new = ast.literal_eval(new_node)
            count = ast.literal_eval(item.value.args[2]) if len(item.value.args) > 2 else -1
            full_cuda = full_cuda.replace(old, new, count)

    mbar = first_literal_assignment("MBAR_HELPERS")
    esc_begin = mbar.index("__device__ __forceinline__ float esc_delta")
    esc_end = mbar.index("// wait with a suspend-time hint", esc_begin)
    cg_begin = full_cuda.index("// ============================================================================ cgemv")
    cg_end = full_cuda.index("// BF16 GEMV", cg_begin)
    source = (
        "#include <ATen/core/Tensor.h>\n"
        "#include <ATen/cuda/CUDAContext.h>\n"
        "#include <c10/cuda/CUDAException.h>\n"
        "#include <cuda_bf16.h>\n#include <stdint.h>\n\n"
        + mbar[esc_begin:esc_end] + full_cuda[cg_begin:cg_end]
    )
    cpp_source = (
        "void cgemv(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
        "int64_t,torch::Tensor,int64_t,int64_t,int64_t);\n"
        "int64_t cgemv_info(int64_t);"
    )
    binding_functions = ["cgemv", "cgemv_info"]
    if "--cgemv-v106" in sys.argv:
        source += first_literal_assignment("_V106_CG_CUDA")
        cpp_source += (
            "\nvoid cgemv_pair(torch::Tensor,torch::Tensor,torch::Tensor,"
            "int64_t,torch::Tensor,int64_t,int64_t);"
            "\nint64_t cgemv_pair_info(int64_t);"
        )
        binding_functions += ["cgemv_pair", "cgemv_pair_info"]
elif "--skinny" in sys.argv:
    # Output-row-parallel sliced-unpooler diagnostic.  This kernel is generic
    # CUDA (no SM90-only instructions), so it can also be correctness-tested
    # on the local Ampere development GPU before the H200 gate.
    source = ast.literal_eval(assignments["_SKINNY_U_CUDA"])
    cpp_source = ast.literal_eval(assignments["_SKINNY_U_CPP"])
    binding_functions = ["skinny_unpool_run"]
elif "--skinny-c12" in sys.argv:
    # Exact row-major C12 m=1 unpooler.  Like the BF16 diagnostic, this is
    # generic CUDA and can be compiled/tested on the local Ampere GPU.
    source = ast.literal_eval(assignments["_SKINNY_C12_CUDA"])
    cpp_source = ast.literal_eval(assignments["_SKINNY_C12_CPP"])
    binding_functions = ["skinny_c12_run"]
elif "--c11-probe" in sys.argv:
    source = ast.literal_eval(assignments["_C11_PROBE_CUDA"])
    cpp_source = ast.literal_eval(assignments["_C11_PROBE_CPP"])
    binding_functions = ["c11_probe_run", "c11_probe_info"]
elif "--gqx-k128-v177" in sys.argv:
    # Isolate v177's production K64 GQX and candidate K128 GQK kernels.  The
    # prefix ends before unrelated later composites, so this validates the
    # exact generated Hopper source without importing SGLang or running any
    # preparation code on the local Ampere GPU.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    prefix, separator, _ = source_text.partition("\nRGEMV_SRC = r")
    if not separator:
        raise RuntimeError("RGEMV source marker not found")
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(prefix, str(source_path), "exec"), namespace)
    common = namespace["CUDA_SRC"].replace(
        "@@MBAR@@", namespace["MBAR_HELPERS"]
    ).replace("@@BODY@@", namespace["WG_HELPERS"])
    source = (
        common
        + namespace["wgmma_ss_defs"]([128, 256])
        + namespace["MAP_SRC"]
        + namespace["_gqx_src"](True)
    )
    cpp_source = (
        "void gqx(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,"
        "int64_t,int64_t,int64_t,int64_t,int64_t);\n"
        "void gqk(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,"
        "int64_t,int64_t,int64_t,int64_t,int64_t);"
    )
    binding_functions = ["gqx", "gqk"]
elif "--rs128-v150" in sys.argv:
    # Isolate v150's faithful chenwainuo RS128 translation unit.  The
    # generated candidate stores the exact CUDA source as a compressed payload
    # so this branch can compile it without invoking prepare() or SGLang.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    source = namespace["_v150_cuda_source"]()
    source = source.replace(
        'cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q)',
        'cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q)',
    )
    cpp_source = namespace["_V150_CPP"]
    binding_functions = ["rqrun"]
elif "--r128-v149" in sys.argv:
    # Isolate v149's two row-parallel register-A warpgroups.  The candidate
    # keeps the production module lazy, so executing definitions here only
    # assembles the exact CUDA translation unit used by the H200 path.
    mock_names = [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ]
    mocks = {name: types.ModuleType(name) for name in mock_names}
    mocks[mock_names[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
    sys.modules.update(mocks)
    namespace = {
        "__name__": "_local_submission_compile_",
        "__file__": str(source_path),
    }
    exec(compile(source_text, str(source_path), "exec"), namespace)
    instances = []
    for inst, _split in namespace["_V149_CONFIGS"].values():
        if inst not in instances:
            instances.append(inst)
    source = namespace["_u32_cuda_source"](instances)
    cpp_source = namespace["_u32_CPP"]
    binding_functions = ["launch", "info"]
elif "--uwide" in sys.argv:
    # v55's wide-C12 experiment reuses the U32 generator with a BN64
    # specialization that is intentionally separate from the production U32
    # module.  Build both the compressed route and its exact BF16 twin.
    namespace = {"__name__": "_local_submission_compile_"}
    exec(compile(source_text, str(source_path), "exec"), namespace)
    configs = namespace["_UWIDE_CONFIGS"]
    instances = []
    for inst, _split in configs.values():
        for compressed in (inst[5], False):
            twin = inst[:5] + (compressed,) + inst[6:]
            if twin not in instances:
                instances.append(twin)
    source = namespace["_u32_cuda_source"](instances)
    cpp_source = namespace["_u32_CPP"]
    binding_functions = ["launch", "info"]
elif "--u32" in sys.argv:
    # The U32 source combines a compressed CUDA core with generated WGMMA
    # wrappers and launch cases.  Evaluating definitions is safe: extensions
    # are loaded lazily by prepare(), which is not called here.
    namespace = {"__name__": "_local_submission_compile_"}
    exec(compile(source_text, str(source_path), "exec"), namespace)
    source = namespace["_u32_cuda_source"](namespace["_u32_INSTANCES"])
    cpp_source = namespace["_u32_CPP"]
    binding_functions = ["launch", "info"]
elif "--u16" in sys.argv or "--p32" in sys.argv:
    payload = assignments["_u16_CPSSM_CUDA_TEMPLATE"].args[0]
    source = zlib.decompress(base64.b64decode(ast.literal_eval(payload))).decode()
    if "--p32" in sys.argv:
        # v41's dormant pooler-32 CPSSM entry after _u16_prepare_cps expands
        # the compact table tuple with fixed-M, matrix shape and geometry.
        inst = (1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1)
        depth = 1
    else:
        inst = ast.literal_eval(assignments["_u16_INST"])
        depth = 2
    instances = [inst, (0,) + inst[1:]]
    cases = []
    for i, instance in enumerate(instances):
        full = instance if len(instance) == 14 else instance + (0,) * 7
        args = ", ".join(map(str, (*full, depth)))
        cases.append(
            f"case {i}: return dd_cap2_b_cpssm_launch_t<{args}>"
            "(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);"
        )
    source = source.replace("@@CASES@@", "\n".join(cases))
    cpp_source = ast.literal_eval(assignments["_u16_CPSSM_CPP"])
    binding_functions = ["dd_cap2_b_cpssm_launch"]
elif "--o32" in sys.argv:
    payload = assignments["_o32_OLAYER_CUDA"].args[0]
    source = zlib.decompress(base64.b64decode(ast.literal_eval(payload))).decode()
    source = source.replace(
        'cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q)',
        'cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",(void**)&encode,cudaEnableDefault,&q)',
    )
    configs = ast.literal_eval(assignments["_o32_CONFIGS"])
    ns = sorted({config[0] for config in configs})
    functions = []
    for n in ns:
        regs = ",".join(f"%{i}" for i in range(n // 2))
        outs = ",".join(f'"+f"(a[{i}])' for i in range(n // 2))
        functions.append(
            f"OD void r3_o32_o_direct_a_ol_mma{n}(float* a,uint64_t w,uint64_t x) {{\n"
            f'  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 '
            f"{{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;\" : {outs} : \"l\"(w),\"l\"(x));\n"
            "}"
        )
    cases = "\n".join(
        f"case {i}: return r3_o32_o_direct_a_ol_launch_t<"
        f"{','.join(map(str, tuple(config) + (1,) if len(config) == 6 else config))}>"
        "(x,w,out,m,stream,query);"
        for i, config in enumerate(configs)
    )
    calls = "\n".join(
        f"if constexpr(BN=={n}) r3_o32_o_direct_a_ol_mma{n}(a,aw,ax);" for n in ns
    )
    source = source.replace("@@MMA@@", "\n".join(functions)).replace("@@CASES@@", cases).replace("@@CALL@@", calls)
    cpp_source = ast.literal_eval(assignments["_o32_OLAYER_CPP"])
    binding_functions = ["r3_o32_o_direct_a_ol_launch"]
elif ("--v102" in sys.argv or "--v101" in sys.argv or "--v100" in sys.argv or "--v99" in sys.argv or "--v98" in sys.argv or "--v97" in sys.argv or "--v96" in sys.argv or "--v93" in sys.argv or "--q128" in sys.argv or
      "--q256-sweep" in sys.argv or "--q256-independent" in sys.argv):
    if "--v102" in sys.argv or "--v101" in sys.argv or "--v100" in sys.argv or "--v99" in sys.argv or "--v98" in sys.argv or "--v97" in sys.argv or "--v96" in sys.argv:
        alt_source = zlib.decompress(
            base64.b64decode(ast.literal_eval(assignments["_ALT_SRC"]))).decode()
        alt_tree = ast.parse(alt_source)
        alt_assignments = {
            target.id: node.value
            for node in alt_tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        payload = alt_assignments["tt_v14_qkv_QKVBIG_CUDA"].args[0]
        qbig_source = zlib.decompress(
            base64.b64decode(ast.literal_eval(payload))).decode()
        qmma_node = next(
            node for node in alt_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "tt_v14_qkv_qbig_mma"
        )
        qmma_namespace = {}
        exec(
            compile(ast.Module(body=[qmma_node], type_ignores=[]), "_alt_proj", "exec"),
            qmma_namespace,
        )
        if "--v102" in sys.argv:
            generator_name = "_v102_cuda_source"
        elif "--v101" in sys.argv:
            generator_name = "_v101_cuda_source"
        elif "--v100" in sys.argv:
            generator_name = "_v100_cuda_source"
        elif "--v99" in sys.argv:
            generator_name = "_v99_cuda_source"
        else:
            generator_name = "_v96_cuda_source"
        generator_node = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == generator_name
        )
        generator_namespace = {
            "_alt": types.SimpleNamespace(
                tt_v14_qkv_QKVBIG_CUDA=qbig_source,
                tt_v14_qkv_qbig_mma=qmma_namespace["tt_v14_qkv_qbig_mma"],
            )
        }
        exec(
            compile(ast.Module(body=[generator_node], type_ignores=[]), str(source_path), "exec"),
            generator_namespace,
        )
        if "--v102" in sys.argv:
            config_name = "_V102_Q128_CONFIGS"
        elif "--v101" in sys.argv:
            config_name = "_V101_Q128_CONFIGS"
        elif "--v100" in sys.argv:
            config_name = "_V100_Q128_CONFIGS"
        elif "--v99" in sys.argv:
            config_name = "_V99_Q128_CONFIGS"
        else:
            config_name = "_V96_Q128_CONFIGS"
        configs = list(ast.literal_eval(assignments[config_name]))
        source = generator_namespace[generator_name](configs)
    elif "--v93" in sys.argv:
        alt_source = zlib.decompress(
            base64.b64decode(ast.literal_eval(assignments["_ALT_SRC"]))).decode()
        alt_tree = ast.parse(alt_source)
        alt_assignments = {
            target.id: node.value
            for node in alt_tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        payload = alt_assignments["tt_v14_qkv_QKVBIG_CUDA"].args[0]
        source = zlib.decompress(base64.b64decode(ast.literal_eval(payload))).decode()
    else:
        payload = assignments["_q128_QKVBIG_CUDA"].args[0]
        source = zlib.decompress(base64.b64decode(ast.literal_eval(payload))).decode()
    source = source.replace(
        'cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q)',
        'cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",(void**)&encode,cudaEnableDefault,&q)',
    )
    if "--v102" in sys.argv or "--v101" in sys.argv or "--v100" in sys.argv or "--v99" in sys.argv or "--v98" in sys.argv or "--v97" in sys.argv or "--v96" in sys.argv:
        pass
    elif "--v93" in sys.argv:
        configs = list(ast.literal_eval(assignments["_V93_Q128_CONFIGS"]))
    elif "--q256-sweep" in sys.argv or "--q256-independent" in sys.argv:
        if "--q256-sweep" in sys.argv:
            helper = next(
                node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_q128_enable_dual_m"
            )
            helper_ns = {}
            exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source_path), "exec"), helper_ns)
            source = helper_ns["_q128_enable_dual_m"](source)
        config_ns = {}
        config_names = (["_q256_BASE"] if "_q256_BASE" in assignments else []) + ["_q256_EXPERIMENT_CONFIGS"]
        for name in config_names:
            node = next(
                node for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            )
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), config_ns)
        configs = config_ns["_q256_EXPERIMENT_CONFIGS"]
    elif "_q128_CONFIG" in assignments:
        configs = [ast.literal_eval(assignments["_q128_CONFIG"])]
    else:
        configs = ast.literal_eval(assignments["_q128_EXPERIMENT_CONFIGS"])
    ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})

    def qmma(n):
        regs = ",".join(f"%{i}" for i in range(n // 2))
        outs = ",".join(f'"+f"(a[{i}])' for i in range(n // 2))
        return (
            f"__device__ __forceinline__ void q_mma_{n}(float* a,uint64_t x,uint64_t w) {{\n"
            f'  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 '
            f"{{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;\" : {outs} : \"l\"(x),\"l\"(w));\n"
            "}"
        )

    source = source.replace("@@MMA@@", "\n".join(qmma(n) for n in ns))
    source = source.replace(
        "@@CALL@@",
        " ".join(f"if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);" for n in ns),
    )
    source = source.replace(
        "@@SWAP@@",
        " ".join(f"if constexpr(TM=={n})q_mma_{n}(a,dw,dx);" for n in ns),
    )
    source = source.replace(
        "@@CASES@@",
        "\n".join(
            f"case {i}: return qb_launch_t<{','.join(map(str, config))},false>"
            "(w,x,out,m,stream,query,stats);"
            for i, config in enumerate(configs)
        ),
    )
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    namespace = "qb_" + digest
    begin = source.index("#define QD")
    args = "int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats"
    source = (
        source[:begin]
        + f"namespace {namespace} {{\n"
        + source[begin:]
        + f"\n}}\nint qb_launch({args}) {{return {namespace}::qb_launch(id,w,x,out,m,stream,query,stats);}}"
    )
    cpp_source = (
        "#include <pybind11/pybind11.h>\n#include <stdint.h>\n"
        "int qb_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);"
    )
    binding_functions = ["qb_launch"]
elif "--gv" in sys.argv:
    # GVCU's switch is generated from Python tuples, so evaluate the module's
    # definitions without calling prepare() or loading any extension.
    namespace = {"__name__": "_local_submission_compile_"}
    exec(compile(source_text, str(source_path), "exec"), namespace)
    cases = "\n".join(
        "    case %d: return gvl<%s>(p, st);" % (index, ",".join(map(str, entry)))
        for index, entry in enumerate(namespace["GVI"])
    )
    source = namespace["GVCU"].replace("@@CASES@@", cases)
    cpp_source = (
        "int64_t gvrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); "
        "int64_t rrrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor);"
    )
    binding_functions = ["gvrun", "rrrun"]
elif "--rh" in sys.argv:
    source = ast.literal_eval(assignments["RHCU"])
    wrappers = []
    for n in (16, 32, 64, 128):
        regs = n // 2
        outputs = ", ".join(f'"+f"(d[{i}])' for i in range(regs))
        registers = ",".join(f"%{i}" for i in range(regs))
        wrappers.append(f'''__device__ __forceinline__ void rhmma{n}(float* d, uint64_t a, uint64_t b) {{
  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{registers}}}, %{regs}, %{regs + 1}, 1, 1, 1, 0, 0;"
               : {outputs} : "l"(a), "l"(b));
}}
''')
    for n in (32, 128):
        nr = n // 2
        outputs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
        registers = ",".join(f"%{i}" for i in range(nr))
        inputs = ", ".join([*(f'"r"(a[{i}])' for i in range(16)),
                             *(f'"l"(b{i})' for i in range(4))])
        ops = ["wgmma.fence.sync.aligned;"]
        for q in range(4):
            aregs = ",".join(f"%{nr + 4 * q + j}" for j in range(4))
            ops.append(f"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 "
                       f"{{{registers}}}, {{{aregs}}}, %{nr + 16 + q}, 1, 1, 1, 0;")
        ops += ["wgmma.commit_group.sync.aligned;", "wgmma.wait_group.sync.aligned 0;"]
        body = "\\n".join(ops)
        wrappers.append(f'''__device__ __forceinline__ void rhmma{n}r(float* d, const uint32_t* a,
                                                        uint64_t b0, uint64_t b1,
                                                        uint64_t b2, uint64_t b3) {{
  asm volatile("{body}" : {outputs} : {inputs} : "memory");
}}
''')
    source = source.replace("RHFNS", "".join(wrappers))
    has_multicast = "rhrunmulti" in source
    cpp_source = (
        "int64_t rhrun(torch::Tensor, torch::Tensor, torch::Tensor); "
        "int64_t rhrun32(torch::Tensor, torch::Tensor, torch::Tensor); "
        "int64_t rhtail(torch::Tensor, torch::Tensor, torch::Tensor);"
        + (" int64_t rhrunmulti(torch::Tensor, torch::Tensor, torch::Tensor);" if has_multicast else "")
    )
    binding_functions = ["rhrun", "rhrun32", "rhtail"] + (["rhrunmulti"] if has_multicast else [])
elif "_ROUTER_CUDA" in assignments:
    source = ast.literal_eval(assignments["_ROUTER_CUDA"])
else:
    # Build submission.py's generated Hopper TMA/WGMMA extension without
    # importing the submission (the local RTX 3050 cannot execute sm_90a).
    source = ast.literal_eval(assignments["WBCU"])
    wbi = ast.literal_eval(assignments["WBI"])
    wri = ast.literal_eval(assignments["WRI"])
    wpi = ast.literal_eval(assignments["WPI"]) if "WPI" in assignments else []
    ns = sorted({entry[0] for entry in wbi + wri} | ({64} if wpi else set()))
    wrappers = []
    for n in ns:
        regs = n // 2
        outputs = ", ".join(f'"+f"(d[{i}])' for i in range(regs))
        registers = ",".join(f"%{i}" for i in range(regs))
        wrappers.append(f'''__device__ __forceinline__ void wb{n}(float* d, uint64_t da, uint64_t db, int sd) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{regs + 2}, 0;\\n"
               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{registers}}}, %{regs}, %{regs + 1}, p, 1, 1, 0, 0;\\n}}\\n"
               : {outputs} : "l"(da), "l"(db), "r"(sd));
}}
''')
    cases = "\n".join(
        "    case %d: return wbl<%s>(p, xp, wp, wr, st);"
        % (index, ",".join(map(str, entry)))
        for index, entry in enumerate(wbi)
    )
    cases += "\n" + "\n".join(
        "    case %d: return wrl<%s>(p, xp, wp, wr, st);"
        % (100 + index, ",".join(map(str, entry)))
        for index, entry in enumerate(wri)
    )
    cases += "\n" + "\n".join(
        "    case %d: return wpl<%s>(p, xp, wp, wr, st);"
        % (200 + index, ",".join(map(str, entry)))
        for index, entry in enumerate(wpi)
    )
    calls = "".join(
        f"if constexpr (BN == {n}) wb{n}(d, bdesc(xa) + 2 * j, bdesc(wa + h * WT) + 2 * j, (c | j | h) != 0); "
        for n in ns
    )
    source = source.replace("WBFNS", "".join(wrappers))
    source = source.replace(
        "WBN(d, bdesc(xa) + 2 * j, bdesc(wa + h * WT) + 2 * j, (c | j | h) != 0);",
        "{ " + calls + "}",
    ).replace("@@CASES@@", cases)
    cpp_source = (
        "int64_t wbrun(int64_t, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor);"
    )
    binding_functions = ["wbrun"]
cuda_flags = ["-O3"]
if ("--skinny" in sys.argv or "--skinny-c12" in sys.argv
        or "--c11-probe" in sys.argv or "--cgemv-v106" in sys.argv
        or "--sk13-v107" in sys.argv or "--gqm-v108" in sys.argv
        or "--wgmma-v109" in sys.argv or "--router-c12-v111" in sys.argv
        or "--sk2-bf16-v112" in sys.argv or "--sk2-l2-v113" in sys.argv
        or "--sk2-localbase-v114" in sys.argv
        or "--sk2-cl2-v116" in sys.argv
        or "--qkv-direct-v117" in sys.argv
         or "--sk2-rowcorr-v118" in sys.argv
         or "--sk2-prodcorr-v119" in sys.argv
         or "--cgemv-radix-v143" in sys.argv
         or "--r128-v149" in sys.argv
         or "--rs128-v150" in sys.argv
         or "--gqx-k128-v177" in sys.argv):
    cuda_flags.extend(["-lineinfo", "--ptxas-options=-v"])
if "--sm90a-compile" in sys.argv:
    cuda_flags.append("-gencode=arch=compute_90a,code=sm_90a")
    cuda_flags.append("--ptxas-options=-v")
extension = load_inline(
    name="okc_bf16_proj_local_" + hashlib.sha256((source + repr(cuda_flags)).encode()).hexdigest()[:12],
    cpp_sources=cpp_source,
    cuda_sources=source,
    extra_cuda_cflags=cuda_flags,
    extra_ldflags=["cublasLt.lib", "cuda.lib"],
    functions=binding_functions,
    with_cuda=True,
    verbose=True,
)
print(
    "CUDA extension compile OK",
    hasattr(extension, "router_fp32_gemv") or hasattr(extension, "router_run") or hasattr(extension, "wbrun")
    or hasattr(extension, "rhrun") or hasattr(extension, "gvrun") or hasattr(extension, "qb_launch")
    or hasattr(extension, "r3_o32_o_direct_a_ol_launch") or hasattr(extension, "dd_cap2_b_cpssm_launch")
    or hasattr(extension, "launch") or hasattr(extension, "rr_launch") or hasattr(extension, "rl_launch")
    or hasattr(extension, "pm_launch") or hasattr(extension, "gqm")
    or hasattr(extension, "skinny_unpool_run") or hasattr(extension, "skinny_c12_run")
    or hasattr(extension, "c11_probe_run") or hasattr(extension, "cgemv")
    or hasattr(extension, "cgemv_pair") or hasattr(extension, "sk13")
    or hasattr(extension, "rk4") or hasattr(extension, "sk2c12")
    or hasattr(extension, "sk2_bf16") or hasattr(extension, "sk2_cand_info")
    or hasattr(extension, "qkv_direct_info") or hasattr(extension, "cgemv_radix_info")
    or hasattr(extension, "gqk")
    or hasattr(extension, "rqrun"),
)

if "--test-cgemv-v104" in sys.argv:
    assert hasattr(extension, "cgemv") and hasattr(extension, "cgemv_info")
    helper_names = {"cg_compress", "_v104_cg_entries", "_v104_retab_rows4"}
    helper_nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    helper_ns = {"torch": torch}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]),
                 str(source_path), "exec"), helper_ns)
    cg_compress = helper_ns["cg_compress"]
    retab = helper_ns["_v104_retab_rows4"]
    torch.manual_seed(104)
    for name, k, nth, info_id in (
            ("qkv_gate", 8192, 256, 1),
            ("o", 1024, 64, 3),
            ("unpooler", 4096, 128, 5)):
        n = 32
        low = torch.randint(0, 256, (n, k), device="cuda", dtype=torch.int32)
        exp = torch.randint(60, 68, (n, k), device="cuda", dtype=torch.int32)
        sign = torch.randint(0, 2, (n, k), device="cuda", dtype=torch.int32)
        for row, col, value in ((0, 0, 52), (1, k // 3, 76),
                                (2, 2 * k // 3, 54), (3, k - 1, 74)):
            exp[row, col] = value
        bits = low | (exp << 8) | (sign << 15)
        weight = torch.where(bits >= 32768, bits - 65536, bits).to(
            torch.int16).view(torch.bfloat16).contiguous()
        cg2 = cg_compress(weight, 2)
        assert cg2 is not None
        cg4 = retab(cg2)
        assert cg4["exact_retab"]
        x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)
        out2 = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
        out4 = torch.empty_like(out2)
        extension.cgemv(cg2["L"], cg2["C"], x, cg2["tab"], cg2["pmax"],
                        out2, cg2["base4"], 2, nth)
        extension.cgemv(cg4["L"], cg4["C"], x, cg4["tab"], cg4["pmax"],
                        out4, cg4["base4"], 4, nth)
        torch.cuda.synchronize()
        expected = (x.double() @ weight.double().t()).to(torch.bfloat16)
        relative_l2 = (torch.linalg.vector_norm(out4.double() - expected.double())
                       / torch.linalg.vector_norm(expected.double()))
        assert torch.equal(out4, out2) and relative_l2 < 0.005
        for col in (0, k // 3, 2 * k // 3, k - 1):
            one = torch.zeros((1, k), device="cuda", dtype=torch.bfloat16)
            one[0, col] = 1
            extension.cgemv(cg4["L"], cg4["C"], one, cg4["tab"],
                            cg4["pmax"], out4, cg4["base4"], 4, nth)
            torch.cuda.synchronize()
            assert torch.equal(out4[0], weight[:, col])
        packed_info = int(extension.cgemv_info(info_id))
        print(f"cgemv-v104 {name}: bitwise_rows2=True exact_probes=True "
              f"relative_l2={relative_l2.item():.3e} "
              f"registers={packed_info & 1023} "
              f"local_bytes={(packed_info >> 10) & ((1 << 20) - 1)} "
              f"active_ctas_per_sm={packed_info >> 50}")

if "--test-cgemv-v105" in sys.argv:
    assert hasattr(extension, "cgemv") and hasattr(extension, "cgemv_info")
    cg_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "cg_compress"
    )
    helper_ns = {"torch": torch}
    exec(compile(ast.Module(body=[cg_node], type_ignores=[]),
                 str(source_path), "exec"), helper_ns)
    cg_compress = helper_ns["cg_compress"]
    torch.manual_seed(105)
    for name, k, nth_base, nth_candidate, info_id in (
            ("qkv_gate", 8192, 256, 512, 1),
            ("o", 1024, 64, 32, 3),
            ("unpooler", 4096, 128, 256, 5)):
        n = 32
        low = torch.randint(0, 256, (n, k), device="cuda", dtype=torch.int32)
        exp = torch.randint(60, 68, (n, k), device="cuda", dtype=torch.int32)
        sign = torch.randint(0, 2, (n, k), device="cuda", dtype=torch.int32)
        for row, col, value in ((0, 0, 52), (1, k // 3, 76),
                                (2, 2 * k // 3, 54), (3, k - 1, 74)):
            exp[row, col] = value
        bits = low | (exp << 8) | (sign << 15)
        weight = torch.where(bits >= 32768, bits - 65536, bits).to(
            torch.int16).view(torch.bfloat16).contiguous()
        cg = cg_compress(weight, 2)
        assert cg is not None
        x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)
        out_base = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
        out_candidate = torch.empty_like(out_base)
        extension.cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"],
                        out_base, cg["base4"], 2, nth_base)
        extension.cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"],
                        out_candidate, cg["base4"], 2, nth_candidate)
        torch.cuda.synchronize()
        expected = (x.double() @ weight.double().t()).to(torch.bfloat16)
        relative_l2 = (
            torch.linalg.vector_norm(out_candidate.double() - expected.double())
            / torch.linalg.vector_norm(expected.double())
        )
        bitwise_base = bool(torch.equal(out_candidate, out_base))
        assert torch.isfinite(out_candidate).all() and relative_l2 < 0.005
        probe_exact = True
        for col in (0, k // 3, 2 * k // 3, k - 1):
            one = torch.zeros((1, k), device="cuda", dtype=torch.bfloat16)
            one[0, col] = 1
            extension.cgemv(cg["L"], cg["C"], one, cg["tab"], cg["pmax"],
                            out_candidate, cg["base4"], 2, nth_candidate)
            torch.cuda.synchronize()
            probe_exact = probe_exact and bool(
                torch.equal(out_candidate[0], weight[:, col]))
        assert probe_exact
        packed_info = int(extension.cgemv_info(info_id))
        print(f"cgemv-v105 {name}: bitwise_base={bitwise_base} "
              f"exact_probes={probe_exact} relative_l2={relative_l2.item():.3e} "
              f"registers={packed_info & 1023} "
              f"local_bytes={(packed_info >> 10) & ((1 << 20) - 1)} "
              f"active_ctas_per_sm={packed_info >> 50}")

if "--test-cgemv-v106" in sys.argv:
    assert hasattr(extension, "cgemv") and hasattr(extension, "cgemv_pair")
    assert hasattr(extension, "cgemv_pair_info")
    helper_names = {"cg_compress", "_v106_pair_pack"}
    helper_nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    helper_ns = {"torch": torch}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]),
                 str(source_path), "exec"), helper_ns)
    cg_compress = helper_ns["cg_compress"]
    pair_pack = helper_ns["_v106_pair_pack"]
    torch.manual_seed(106)
    for name, k, nth, info_id in (
            ("qkv_gate", 8192, 512, 1),
            ("o", 1024, 64, 2),
            ("pooler", 8192, 512, 1),
            ("unpooler", 4096, 128, 4)):
        n = 32
        low = torch.randint(0, 256, (n, k), device="cuda", dtype=torch.int32)
        exp = torch.randint(60, 68, (n, k), device="cuda", dtype=torch.int32)
        sign = torch.randint(0, 2, (n, k), device="cuda", dtype=torch.int32)
        for row, col, value in ((0, 0, 52), (1, k // 3, 76),
                                (2, 2 * k // 3, 54), (3, k - 1, 74)):
            exp[row, col] = value
        bits = low | (exp << 8) | (sign << 15)
        weight = torch.where(bits >= 32768, bits - 65536, bits).to(
            torch.int16).view(torch.bfloat16).contiguous()
        cg = cg_compress(weight, 2)
        assert cg is not None
        packed, pack_exact = pair_pack(cg)
        assert pack_exact and packed.numel() == cg["L"].numel() + cg["C"].numel()
        x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)
        out_base = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
        out_candidate = torch.empty_like(out_base)
        extension.cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"],
                        out_base, cg["base4"], 2, nth)
        extension.cgemv_pair(packed, x, cg["tab"], cg["pmax"],
                             out_candidate, cg["base4"], nth)
        torch.cuda.synchronize()
        expected = (x.double() @ weight.double().t()).to(torch.bfloat16)
        relative_l2 = (
            torch.linalg.vector_norm(out_candidate.double() - expected.double())
            / torch.linalg.vector_norm(expected.double())
        )
        bitwise_base = bool(torch.equal(out_candidate, out_base))
        assert bitwise_base and torch.isfinite(out_candidate).all()
        assert relative_l2 < 0.005
        probe_exact = True
        for col in (0, k // 3, 2 * k // 3, k - 1):
            one = torch.zeros((1, k), device="cuda", dtype=torch.bfloat16)
            one[0, col] = 1
            extension.cgemv_pair(packed, one, cg["tab"], cg["pmax"],
                                 out_candidate, cg["base4"], nth)
            torch.cuda.synchronize()
            probe_exact = probe_exact and bool(
                torch.equal(out_candidate[0], weight[:, col]))
        assert probe_exact
        packed_info = int(extension.cgemv_pair_info(info_id))
        print(f"cgemv-v106 {name}: pack_exact={pack_exact} "
              f"bitwise_base={bitwise_base} exact_probes={probe_exact} "
              f"relative_l2={relative_l2.item():.3e} "
              f"registers={packed_info & 1023} "
              f"local_bytes={(packed_info >> 10) & ((1 << 20) - 1)} "
              f"active_ctas_per_sm={packed_info >> 50}")

if "--test-skinny" in sys.argv:
    assert hasattr(extension, "skinny_unpool_run")
    torch.manual_seed(79)
    weight = torch.randn((1024, 4096), device="cuda", dtype=torch.bfloat16)
    variants = {1: range(0, 5), 8: range(5, 11)}
    for rows, instances in variants.items():
        x = torch.randn((rows, 4096), device="cuda", dtype=torch.bfloat16)
        expected = (x.double() @ weight.double().t()).to(torch.bfloat16)
        for instance in instances:
            actual = torch.empty((rows, 1024), device="cuda", dtype=torch.bfloat16)
            result = extension.skinny_unpool_run(instance, weight, x, actual)
            torch.cuda.synchronize()
            assert result == 0
            relative_l2 = torch.linalg.vector_norm(actual.double() - expected.double()) / torch.linalg.vector_norm(expected.double())
            assert torch.isfinite(actual).all() and relative_l2 < 0.004
            print(f"skinny m={rows} instance={instance}: relative_l2={relative_l2.item():.3e}")

if "--test-skinny-c12" in sys.argv:
    assert hasattr(extension, "skinny_c12_run")
    namespace = {"__name__": "_local_skinny_c12_test_"}
    exec(compile(source_text, str(source_path), "exec"), namespace)
    torch.manual_seed(81)
    lo = torch.randint(0, 256, (1024, 4096), device="cuda", dtype=torch.int32)
    eh = torch.randint(60, 68, (1024, 4096), device="cuda", dtype=torch.int32)
    eh[:, (13, 997, 2049, 4051)] = torch.tensor((52, 75, 54, 73), device="cuda", dtype=torch.int32)
    sign = torch.randint(0, 2, (1024, 4096), device="cuda", dtype=torch.int32)
    bits = lo | (eh << 8) | (sign << 15)
    weight = torch.where(bits >= 32768, bits - 65536, bits).to(torch.int16).view(torch.bfloat16).contiguous()
    packed = namespace["_skinny_c12_pack"](weight)
    assert packed is not None and packed["max_esc"] == 4
    x = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
    expected = (x.double() @ weight.double().t()).to(torch.bfloat16)
    for instance in (2, 3):
        actual = torch.empty((1, 1024), device="cuda", dtype=torch.bfloat16)
        result = extension.skinny_c12_run(instance, packed["w"], packed["esc"], x, actual, packed["table"])
        torch.cuda.synchronize()
        assert result == 0
        relative_l2 = torch.linalg.vector_norm(actual.double() - expected.double()) / torch.linalg.vector_norm(expected.double())
        assert torch.isfinite(actual).all() and relative_l2 < 0.004
        print(f"skinny-c12 instance={instance}: relative_l2={relative_l2.item():.3e}")
    for col in (0, 13, 996, 997, 2049, 4051, 4095):
        one = torch.zeros((1, 4096), device="cuda", dtype=torch.bfloat16)
        one[0, col] = 32
        actual = torch.empty((1, 1024), device="cuda", dtype=torch.bfloat16)
        assert extension.skinny_c12_run(2, packed["w"], packed["esc"], one, actual, packed["table"]) == 0
        torch.cuda.synchronize()
        assert torch.equal(actual[0], (weight[:, col] * 32).to(torch.bfloat16))
    print("skinny-c12 exact one-hot OK")

if "--test-original-router" in sys.argv:
    assert hasattr(extension, "router_run")
    torch.manual_seed(17)
    weight = torch.randn((256, 8192), device="cuda", dtype=torch.float32)
    weight_hi = weight.to(torch.bfloat16).contiguous()
    weight_lo = (weight - weight_hi.float()).to(torch.bfloat16).contiguous()
    for rows in (1, 8, 16, 32, 128, 256):
        x = torch.randn((rows, 8192), device="cuda", dtype=torch.bfloat16)
        actual = extension.router_run(x, weight, weight_hi, weight_lo)
        expected = x.double() @ weight.double().T
        relative_l2 = torch.linalg.vector_norm(actual.double() - expected) / torch.linalg.vector_norm(expected)
        row_denominator = torch.linalg.vector_norm(expected, dim=1).clamp_min(
            torch.linalg.vector_norm(expected) / rows**0.5 / 10
        )
        worst_row = (
            torch.linalg.vector_norm(actual.double() - expected, dim=1) / row_denominator
        ).max()
        print(
            f"original router m={rows}: relative_l2={relative_l2.item():.3e} "
            f"worst_row={worst_row.item():.3e} finite={bool(torch.isfinite(actual).all())}"
        )

if "--test-original-variants" in sys.argv:
    assert hasattr(extension, "router_run")
    torch.manual_seed(23)
    weight = torch.randn((256, 8192), device="cuda", dtype=torch.float32)
    weight_hi = weight.to(torch.bfloat16).contiguous()
    weight_lo = (weight - weight_hi.float()).to(torch.bfloat16).contiguous()
    variant_sets = {
        1: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16),
        8: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16),
        16: (10, 11, 12, 13, 50, 51, 52, 53),
        32: (10, 11, 12, 13, 20, 21, 22, 50, 51, 52, 53),
        128: (20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 53),
        256: (20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 53),
    }
    for rows, variants in variant_sets.items():
        x = torch.randn((rows, 8192), device="cuda", dtype=torch.bfloat16)
        expected = x.double() @ weight.double().T
        for variant in variants:
            actual = extension.router_run(x, weight, weight_hi, weight_lo, variant)
            relative_l2 = (
                torch.linalg.vector_norm(actual.double() - expected)
                / torch.linalg.vector_norm(expected)
            )
            assert torch.isfinite(actual).all() and relative_l2 < 1e-4
        print(f"original router variants m={rows}: {len(variants)} passed")

if "--bench-original-router" in sys.argv:
    assert hasattr(extension, "router_run")
    torch.manual_seed(19)
    weight = torch.randn((256, 8192), device="cuda", dtype=torch.float32)
    weight_hi = weight.to(torch.bfloat16).contiguous()
    weight_lo = (weight - weight_hi.float()).to(torch.bfloat16).contiguous()
    flush_source = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
    flush_destination = torch.empty_like(flush_source)
    for rows in (1, 8, 16, 32, 128, 256):
        x = torch.randn((rows, 8192), device="cuda", dtype=torch.bfloat16)
        extension.router_run(x, weight, weight_hi, weight_lo)
        torch.cuda.synchronize()
        samples = []
        for _ in range(20):
            flush_destination.copy_(flush_source)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            extension.router_run(x, weight, weight_hi, weight_lo)
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) * 1000)
        samples.sort()
        print(
            f"original router m={rows}: cold_median_us={samples[len(samples)//2]:.3f} "
            f"cold_best_us={samples[0]:.3f}"
        )

if "--bench-original-variants" in sys.argv:
    assert hasattr(extension, "router_run")
    torch.manual_seed(29)
    weight = torch.randn((256, 8192), device="cuda", dtype=torch.float32)
    weight_hi = weight.to(torch.bfloat16).contiguous()
    weight_lo = (weight - weight_hi.float()).to(torch.bfloat16).contiguous()
    flush_source = torch.empty(20 << 20, device="cuda", dtype=torch.float32)
    flush_destination = torch.empty_like(flush_source)
    for rows in (1, 8):
        x = torch.randn((rows, 8192), device="cuda", dtype=torch.bfloat16)
        results = []
        for variant in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16):
            extension.router_run(x, weight, weight_hi, weight_lo, variant)
            torch.cuda.synchronize()
            samples = []
            for _ in range(7):
                torch.add(flush_source, 1.0, out=flush_destination)
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                extension.router_run(x, weight, weight_hi, weight_lo, variant)
                stop.record()
                stop.synchronize()
                samples.append(start.elapsed_time(stop) * 1000.0)
            samples.sort()
            results.append((samples[len(samples) // 2], variant))
        print(f"original router variant bench m={rows}: {sorted(results)}")

if "--test-router" in sys.argv:
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    weight = torch.randn((256, 8192), device="cuda", dtype=torch.float32)
    for rows in range(1, 9):
        x = torch.randn((rows, 8192), device="cuda", dtype=torch.bfloat16)
        actual = extension.router_fp32_gemv(x, weight)
        expected = torch.mm(x.float(), weight.t())
        relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
        print(
            f"router m={rows}: relative_l2={relative_l2.item():.3e} "
            f"max_abs={(actual - expected).abs().max().item():.3e}"
        )

if "--bench-router" in sys.argv:
    torch.manual_seed(2)
    weight = torch.randn((256, 8192), device="cuda", dtype=torch.float32)
    flush_source = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
    flush_destination = torch.empty_like(flush_source)
    for rows in range(1, 9):
        x = torch.randn((rows, 8192), device="cuda", dtype=torch.bfloat16)
        extension.router_fp32_gemv(x, weight)
        torch.cuda.synchronize()
        samples = []
        for _ in range(30):
            flush_destination.copy_(flush_source)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            extension.router_fp32_gemv(x, weight)
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) * 1000.0)
        samples.sort()
        print(
            f"router m={rows}: cold_median_us={samples[len(samples) // 2]:.3f} "
            f"cold_best_us={samples[0]:.3f}"
        )
