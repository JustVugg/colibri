@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d C:\Users\Mark\Desktop\Projects\colibri\c
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc" -O3 -std=c++17 -arch=sm_120 -Xcompiler=-W3 -shared -DCOLI_CUDA_BUILDING_DLL -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib/x64" -lcudart backend_cuda.cu -o coli_cuda.dll
echo EXITCODE=%ERRORLEVEL%
