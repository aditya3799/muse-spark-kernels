$ErrorActionPreference = "Stop"
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$configured = & cmd.exe /d /s /c "`"$vcvars`" >nul && set"
foreach ($line in $configured) {
    $separator = $line.IndexOf("=")
    if ($separator -gt 0) {
        $name = $line.Substring(0, $separator)
        $value = $line.Substring($separator + 1)
        Set-Item -LiteralPath "Env:$name" -Value $value
    }
}
$env:TORCH_CUDA_ARCH_LIST = "8.6"
$env:TORCH_EXTENSIONS_DIR = Join-Path $PSScriptRoot ".torch_extensions_codec_e_gemv"
& "C:\Users\Aditya Kudale\AppData\Local\Programs\Python\Python310\python.exe" `
    -m pytest -q -p no:cacheprovider (Join-Path $PSScriptRoot "test_codec_e_gemv_probe.py")
exit $LASTEXITCODE
