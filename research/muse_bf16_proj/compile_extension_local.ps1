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
$env:TORCH_CUDA_ARCH_LIST = if ($args -contains "--sm90a-compile") { "9.0a" } else { "8.6" }
& "C:\Users\Aditya Kudale\AppData\Local\Programs\Python\Python310\python.exe" compile_extension_local.py @args
exit $LASTEXITCODE
