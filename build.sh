
echo "NOTE: Build takes a lot of time. "
echo "NOTE: In tool calling, timeout is given as milliseconds (1000 = 1 second, 60000 = 1 minute, 3600000 = 1 hour). Specify a long timeout like 3600000 ms (1 hour)"

nix develop .#cuda --command bash -c 'cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_RPATH=/run/opengl-driver/lib -DCMAKE_INSTALL_RPATH=/run/opengl-driver/lib -GNinja -DGGML_CUDA_FA_PREFILL_F16=N -DGGML_CUDA_FA_QUANTS="pq2_0-pq2_0;q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16" && cmake --build build --parallel 12'
