@echo off
rem Experimental standalone vision build. Uses the same private ROCm SDK as tools/hip/build_windows.bat.
rem Prerequisites: pinned llama.cpp in third_party/llama.cpp, VS C++ tools, CMake, .rocm-win SDK initialized.
setlocal
for %%I in ("%~dp0..\..") do set "SRC=%%~fI"
if not defined ROCM_VENV set "ROCM_VENV=%SRC%\.rocm-win"
if not defined STRATA_HIP_ARCHS set "STRATA_HIP_ARCHS=gfx1100"
if not exist "%ROCM_VENV%\Scripts\rocm-sdk.exe" (echo Install the private ROCm SDK first & exit /b 1)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS="
for /f "delims=" %%V in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VS=%%V"
if not defined VS (echo Visual Studio C++ Build Tools are required & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "PATH=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
for /f "delims=" %%R in ('"%ROCM_VENV%\Scripts\rocm-sdk.exe" path --root') do set "ROCM=%%R"
if not defined ROCM (echo Initialize the private ROCm SDK first & exit /b 1)
set "ROCM_F=%ROCM:\=/%"
set "ROCM_PATH=%ROCM%"
set "HIP_PATH=%ROCM%"
set "HIP_PLATFORM=amd"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
cmake -S "%SRC%\tools\vision" -B "%SRC%\build-vision-hip" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_VISION_HIP=ON -DSTRATA_VISION_CUDA=OFF ^
  -DSTRATA_PORTABLE=ON "-DLLAMA_DIR=%SRC%/third_party/llama.cpp" ^
  "-DCMAKE_C_COMPILER=%ROCM_F%/lib/llvm/bin/clang.exe" ^
  "-DCMAKE_CXX_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" ^
  "-DCMAKE_PREFIX_PATH=%ROCM_F%" "-DGPU_TARGETS=%STRATA_HIP_ARCHS%" ^
  "-DCMAKE_CXX_FLAGS=--rocm-path=%ROCM_F% --rocm-device-lib-path=%ROCM_F%/lib/llvm/amdgcn/bitcode" || exit /b 1
cmake --build "%SRC%\build-vision-hip" --target strata-vision --parallel 4 || exit /b 1
rem Keep the SDK runtime ahead of AMD driver copies in System32, as the main HIP engine does.
for %%D in (amdhip64_7.dll amd_comgr.dll) do (
  copy /y "%ROCM%\bin\%%D" "%SRC%\build-vision-hip\bin\%%D" >nul
  if errorlevel 1 exit /b 1
)
echo Encoder built in %SRC%\build-vision-hip\bin
endlocal
