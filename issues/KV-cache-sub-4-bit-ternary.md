# Issue: sub-4-bit (1-bit ternary / 2-bit) KV cache

Status: WRITE PATH DONE, ATTENTION READ PATH BROKEN (see "BUG" + "DECISION" sections).
Unit test `test-kv-mean-center` passes on CUDA0 (RTX 2080 Ti): all 3 subcases OK,
nmse(baseline, centered) = 3.03e-16. K/V *writes* now run on CUDA (SET_ROWS + cpy accept
PQ2_0/PTQ1_0). But the FA *read* path does NOT register ptq_0/pq2_0, so with
`-ctk ptq_0 -ctv ptq_0 -fa on` the whole attention op offloads to the CPU -> 10x slow,
GPU util 10-20% (q4_0 baseline 99-100%). Fix planned in two commits (DECISION). All edits
on branch `umberto`.

## Goal

Enable 1-bit ternary (PTQ1_0) KV cache for Bonsai 2 27B on CUDA, i.e.
`-ctk ptq_0 -ctv ptq_0` and `-ctk pq2_0 -ctv pq2_0` in `llama server` / `llama-cli`.
Paper (bonsai-paper.txt, Section 5) states: "KV compression (sub-4-bit). Early results show
KV cache can be pushed toward the sub-4-bit regime, while retaining benchmark performance."

## Findings

### Existing PTQ1_0/PQ2_0 primitives (all present in repo, CUDA included)
- `ggml/src/ggml-cuda/ggml-cuda.cu` dispatch:
  - `GGML_OP_MUL_MAT` / `MUL_MAT_ID`: PTQ1_0 + PQ2_0 supported (line ~5657)
  - `GGML_OP_GET_ROWS`: PTQ1_0 + PQ2_0 supported (line ~5705)
  - `GGML_OP_SET_ROWS`: now supports PQ2_0/PTQ1_0 (added in commit 8525ab8e9)
  - `GGML_OP_CPY` (via f32<->ptq_0/pq2_0 entries): now supported (commit 28ffe0d36)
- `ggml/src/ggml-cuda/mmq.cu` / `ggml/src/ggml-cuda/mmvq.cu` /
  `ggml/src/ggml-cuda/getrows.cu`: working `mul_mat` / `mul_mat_vec` / `get_rows` for
  PTQ1_0 and PQ2_0
- `ggml/src/ggml-cuda/convert.cu`: `dequantize_row_ptq1_0_cuda` + `dequantize_row_pq2_0`
  + generic `dequantize_block_cuda`
- `ggml/src/ggml-cpu/quants.c` + `ggml/src/ggml-cpu/ggml-cpu.cpp`: full
  `quantize_row_ptq1_0` + `dequantize_row_ptq1_0` + CPU SIMD dequant
- SYCL also has it but out of scope for this issue

### Original gaps (now mostly closed)
1. CUDA copy (write) path: NO f32<->PTQ1_0/PQ2_0 entry in cpy.cu; `llama_kv_cache::cpy_k`
   (src/llama-kv-cache.cpp:1366) aborted at `GGML_ABORT("unsupported type combination")`.
   DONE: f32<->PTQ1_0 / f32<->PQ2_0 entries added (cpu bit-for-bit quantize ports).
2. `SET_ROWS` dispatch/assert: blocked quantized-dst for ptq_0/pq2_0. DONE (see
   "RESOLVED (write path)").
3. Mean centering gate: Q4_0-only. DONE: generalized to {Q4_0, PQ2_0, PTQ1_0}.
4. arg.cpp kv_cache_types: missing PTQ1_0/PQ2_0. DONE.
5. Flash attention: NOT supported (the current BUG). See "BUG" + "DECISION".

## What is already working (baseline)
- `-ctk q4_0 -ctv q4_0` works on CUDA
- `-ctk ptq_0 -ctv ptq_0` / `-ctk pq2_0 -ctv pq2_0`: K/V *writes* run on CUDA (no crash),
  but decode attention falls back to CPU (see BUG)
- `--kv-mean-center` works with q4_0 (docs/kv-mean-center.md, tools/kv-mean-center/)
- Hadamard K-rotation for quantized K is automatic (llama-kv-cache.cpp:325-339)
- Current command that works (q4_0 baseline, 99-100% GPU):
  ```
  build/bin/llama serve -np 1 --host 0.0.0.0 \
    -hf prism-ml/Ternary-Bonsai-2-27B-gguf:TQ1_0 \
    --jinja -ngl 99 -kvu \
    -c 131072 \
    -fa on \
    -ctk q4_0 \
    -ctv q4_0 \
    --port 8080 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --reasoning-budget 8192 --reasoning-preserve
  ```

## Commits
- `28ffe0d36` kv-cache: enable sub-4-bit (ptq_0/pq2_0) caches on CUDA (write path)
- `8525ab8e9` kv-cache: fix SET_ROWS pre-allocation abort for ptq_0/pq2_0 on CUDA;
  generalize mean-center gate + test

## Build / test plan
Build (user's command):
  nix develop .#cuda --command bash -c 'cmake -B build -DGGML_CUDA=ON
    -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_RPATH=/run/opengl-driver/lib
    -DCMAKE_INSTALL_RPATH=/run/opengl-driver/lib -GNinja
    -DGGML_CUDA_FA_PREFILL_F16=N && cmake --build build --parallel'
Unit test (passes, CUDA0):
  ./build/bin/test-kv-mean-center   # all 3 subcases OK, nmse=3.03e-16, EXIT 0

## Implementation (write path, DONE)
1. `ggml/src/ggml-cuda/cpy-utils.cuh` - `quantize_f32_ptq1_0_block` +
   `quantize_f32_pq2_0_block` (bit-for-bit ports) + wrappers.
2. `ggml/src/ggml-cuda/set-rows.cu` - quantized dispatch:
   `set_rows_cuda_quant<idx_t, block_pq2_0, QK_PQ2_0, quantize_f32_pq2_0_block>` and
   `set_rows_cuda_quant<idx_t, block_ptq_0, QK_PTQ1_0, quantize_f32_ptq1_0_block>`.
3. `ggml/src/ggml-cuda/cpy.cu` - `cpy_blck_ptq1_0_f32` / `cpy_blck_pq2_0_f32` + four
   global launch funcs + dispatch branches F32<->PTQ1_0 / F32<->PQ2_0.
4. `common/arg.cpp` - kv_cache_types now includes PTQ1_0/PQ2_0 (`ptq_0` / `pq2_0`).
5. `src/llama-context.cpp` - mean-center gate generalized to {Q4_0, PQ2_0, PTQ1_0}.
6. `src/llama-kv-cache.cpp/.h` - `load_kv_mean_center` type-agnostic; gate moved to
   public path. `llama-context.cpp` public gate added (8525ab8e9).

## RESOLVED (write path) -- SET_ROWS / cpy
The K/V *write* path works: SET_ROWS + cpy dequant entries accept PQ2_0/PTQ1_0.

## RESOLVED: SET_ROWS crash
Three gaps fixed:
1. Scheduler gate -- `ggml_backend_cuda_device_supports_op` (ggml-cuda.cu) did NOT list
   PQ2_0/PTQ1_0 in the SET_ROWS quantized-dst list. Fix: added both types.
2. Kernel entry assert -- `ggml_cuda_op_set_rows` (set-rows.cu) required
   `src0 F32 OR (src0 F16 AND dst F16)`. Fix: relax to `src0 F32 OR src0 F16`.
3. Test seam -- `llama_kv_cache::load_kv_mean_center` rejected F32 K cache. Fix: type-
   agnostic; gate moved to public `--kv-mean-center` path.
Verified: `build/bin/test-kv-mean-center` -> all 3 subcases OK, EXIT 0, nmse=3.03e-16.

## BUG: attention offloads to CPU for ptq_0/pq2_0 (2026-10-16)
Observed (user, RTX 2080 Ti, `-ctk ptq_0 -ctv ptq_0 -fa on`): model runs, no crash, BUT
throughput ~10x lower, CUDA util ~10-20% (q4_0 baseline 99-100%), decode on CPU.
Root cause: the FA *read* path never registers ptq_0/pq2_0, so the whole attention op
offloads to CPU.
Chain:
- `ggml_cuda_fattn_kv_type_supported` (fattn.cu:374-388) accepts only
  F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 -- NOT PQ2_0/PTQ1_0.
- `ggml_cuda_get_best_fattn_kernel` (fattn.cu:474) -> `BEST_FATTN_KERNEL_NONE`.
- `ggml_cuda_flash_attn_ext_supported` (fattn.cu:627) -> false.
- `ggml_backend_cuda_device_supports_op` (ggml-cuda.cu:6013) -> false.
- llama.cpp backend -> attention op offloaded to CPU (GPU only does the weight matmuls).
Secondary: `ggml_cuda_fattn_mma_kv_native` (fattn-mma-f16.cuh:366) and its dispatch
(fattn.cu:148-153, 176-181) only have in-place quantized kernels for Q4_0/Q8_0; any other
quantized K/V type that slipped past the selector would hit the `else -> Q8_0` branch and
misread the cache. So both the gate AND the native kernel must be extended.
Note: `ggml_is_quantized(PQ2_0/PTQ1_0)` is true (ggml.c:696/704), so the selector already
routes them to the quantized branch, but the gate rejects them at line 474.

## DECISION (2026-10-16)
Proceed in two small commits (keep each reviewable):
  STEP 1 (now, cheap): make **pq2_0** attention run on CUDA.
    - Add `GGML_TYPE_PQ2_0` to `ggml_cuda_fattn_kv_type_supported` (fattn.cu:374-388)
      and `ggml_cuda_fattn_mma_kv_native` (fattn-mma-f16.cuh:365-367).
    - Extend the dispatch in `ggml_cuda_flash_attn_ext_mma_f16` (fattn.cu ne[0]==128/256
      cases, currently an `if Q4_0 else Q8_0` -> add an explicit PQ2_0 arm; fix the
      else-to-Q8_0 trap).
    - Add a dequant tile loader for PQ2_0 (reuse `dequantize_row_pq2_0`; QK_PQ2_0 = 64
      weights per block, same tiling as Q4_0, so it largely mirrors the Q4_0 case).
    - Instantiate via `DECL_FATTN_MMA_F16_CASE_KV_ALL_NCOLS2(..., GGML_TYPE_PQ2_0)`.
  STEP 2 (later): make **ptq_0** attention run on CUDA.
    - PTQ1_0 needs its own loader + staging: 4-bit packed (differs from Q4_0's 2-bit) and
      QK_PTQ1_0 = 128 weights per block (Q4_0 is 32), so it cannot reuse Q4_0 tiling.
    - Mirror the existing Q4_0/Q8_0 pattern; instantiate for PTQ1_0.
Fix components (both steps):
- gate: `ggml_cuda_fattn_kv_type_supported` (fattn.cu:374-388)
- native flag: `ggml_cuda_fattn_mma_kv_native` (fattn-mma-f16.cuh:365-367)
- dispatch: `ggml_cuda_flash_attn_ext_mma_f16` (fattn.cu ne[0]==128/256 cases)
- dequant tile loader + kernel instantiations (DECL_FATTN_MMA_F16_CASE_KV_ALL_NCOLS2)
Verify each step with `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` during
decode: util must return to ~99% (not 10-20%).

## OPEN
- No precomputed mean-centering bias calibrated for ptq_0 / pq2_0 exists.
  `tools/kv-mean-center/` currently only produces one for q4_0. So `--kv-mean-center` with
  a q4_0 bias on ptq_0 would be a basis mismatch (checked via `kv_mean_center.k_rot` gguf
  key). The NON-mean-centering path is what is enabled now.
- FA in-place dequant kernel for PQ2_0 (step 1) + PTQ1_0 (step 2): not yet written
  (see BUG + DECISION).

## VERIFY
Build (above), then:
  Test A (pq2_0, after step 1):
    ./build/bin/llama server -np 1 --host 0.0.0.0
      -hf prism-ml/Ternary-Bonsai-2-27B-gguf:TQ1_0 --jinja -ngl 99 -kvu
      -c 131072 -fa on -ctk pq2_0 -ctv pq2_0 --port 8080
      --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00
      --reasoning-budget 8192 --reasoning-preserve
    hit http://localhost:8080/v1/completions; check sensible output AND
    nvidia-smi decode utilization ~99% (GPU, not CPU).
  Test B (ptq_0, after step 2): same but -ctk ptq_0 -ctv ptq_0.
  Baseline (q4_0): already confirmed 99-100% GPU, sensible output.
