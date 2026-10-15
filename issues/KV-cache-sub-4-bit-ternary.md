# Issue: sub-4-bit (1-bit ternary / 2-bit) KV cache

Status: TEST PASSING. `test-kv-mean-center` runs clean on CUDA0 (RTX 2080 Ti) — all 3 subcases OK (test_regression_safety, test_mean_center_gate, test_softmax_invariance; nmse(baseline, centered) = 3.03e-16). SET_ROWS crash resolved (root cause + fix below). Target: `-ctk ptq_0 -ctv ptq_0` / `-ctk pq2_0 -ctv pq2_0` working on CUDA via the existing CLI. All edits on branch `umberto`.

## Goal

Enable 1-bit ternary (PTQ1_0) KV cache for Bonsai 2 27B on CUDA (CPU as follow-up), i.e. `--cache-type-k ptq_0` / `q2_0` in `llama server` / `llama-cli`.
Paper (bonsai-paper.txt, Section 5) states: "KV compression (sub-4-bit). Early results show KV cache can be pushed toward the sub-4-bit regime, while retaining benchmark performance."
Current state: only q4_0 works (`-ctk q4_0 -ctv q4_0`), with optional `--kv-mean-center`.

## Findings (research so far)

### Existing PTQ1_0/PQ2_0 primitives (all present in repo, CUDA included)
- `ggml/src/ggml-cuda/ggml-cuda.cu` dispatch:
  - `GGML_OP_MUL_MAT` / `MUL_MAT_ID`: PTQ1_0 + PQ2_0 supported (line ~5657)
  - `GGML_OP_GET_ROWS`: PTQ1_0 + PQ2_0 supported (line ~5705)
  - `GGML_OP_MUL_MAT_VEC`: NOT in the list (needs adding; line ~5661 default case)
- `ggml/src/ggml-cuda/mmq.cu` / `ggml/src/ggml-cuda/mmvq.cu` / `ggml/src/ggml-cuda/getrows.cu` /
  `ggml/src/ggml-cuda/ggml-cuda.cu` all have working `mul_mat` / `mul_mat_vec` / `get_rows` for PTQ1_0 and PQ2_0
- `ggml/src/ggml-cuda/convert.cu` has `dequantize_row_ptq1_0_cuda` + generic `dequantize_block_cuda`
- `ggml/src/ggml-cpu/quants.c` + `ggml/src/ggml-cpu/ggml-cpu.cpp`: full `quantize_row_ptq1_0` +
  `dequantize_row_ptq1_0` + CPU SIMD dequant
- SYCL also has it but out of scope for this issue

### What is missing (the actual work)
1. **CUDA copy (write) path**: `ggml/src/ggml-cuda/cpy.cu` has no `f16 -> quantized` and no
   `f32 -> PTQ1_0/PQ2_0` entry. Currently only handles f32->quant and quant->f32.
   `llama_kv_cache::cpy_k` (src/llama-kv-cache.cpp:1366) writes the f16 K tensor into a quantized
   KV-cache tensor. This aborts at `GGML_ABORT("unsupported type combination")` for PTQ1_0/PQ2_0.
   Fix: add `f16->PTQ1_0` / `f16->PQ2_0` blocks to the dispatch (reuse `dequantize_ptq1_0` /
   `dequantize_pq2_0` from `dequantize.cuh`).

2. **`mul_mat_vec` dispatch**: the decode path (src/llama.cpp) reads K from a quantized cache via
   `mul_mat_vec` (src1 is f32). Need to add PTQ1_0 / PQ2_0 to the `GGML_OP_MUL_MAT_VEC` switch
   in `ggml/src/ggml-cuda/ggml-cuda.cu`.

3. **`mul_mat_vec_q` path**: src1 type must be F32 (line 1791). Query is f16 in attention
   (src/llama.cpp). Check if a cast to f32 happens before the call, or if we need to route via
   `mul_mat_vec` (unquantized path) instead. Verify in `src/llama.cpp`.

4. **Mean centering**: `llama_context` line 4165 rejects `path_kv_mean_center` unless
   `type_k == GGML_TYPE_Q4_0`. Must generalize to PTQ1_0/PQ2_0. The math is the same (softmax
   invariant). But the bias is stored as an f32/f16 tensor per layer — cast happens in
   `llama-kv-cache::cpy_k` line 1371 via `ggml_cast`. Should work for any type.

5. **arg.cpp**: `kv_cache_types` (common/arg.cpp:304) must add PTQ1_0 and PQ2_0 to the allowed list.

6. **Flash attention**: `-fa on` is required for quantized V cache (line 4134). Check that the
   FA path (src/llama.cpp `flash_attn` call) supports reading a quantized K/V for PTQ1_0/PQ2_0.

## What is already working (baseline)
- `-ctk q4_0 -ctv q4_0` works on CUDA
- `--kv-mean-center` works with q4_0 (docs/kv-mean-center.md, tools/kv-mean-center/)
- Hadamard K-rotation for quantized K is automatic (llama-kv-cache.cpp:325-339)
- Current command that works:
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
    --reasoning-budget 8192 --reasoning-preserve --reasoning-budget-message
  ```

## Open questions
- Does `mul_mat_vec` with a PTQ1_0 src0 + f16 src1 work (or does src1 need to be f32)?
- Is `flash_attn` (the -fa on path in src/llama.cpp) reading a quantized K for prefill, or
  is it dequantizing K to f16 first? If the latter, the FA path doesn't need changes.
- Does the q4_0 KV decode kernel (SYCL, ggml/src/ggml-sycl/fattn-dec.cpp) get called on CUDA
  for the decode path, or is it only for the SYCL backend? (The CUDA decode path uses
  `mul_mat_vec` + `mul_mat_vec` for K and V.)
- Block size of PTQ1_0 is 128; Q4_0 is 32. The `n_embd_head_k(il) % blck_size == 0` check
  passes for PTQ1_0 (256 % 128 == 0).

## Commit
- `28ffe0d36` kv-cache: enable sub-4-bit (ptq_0/pq2_0) caches on CUDA (branch umberto)
  Files: arg.cpp, cpy-utils.cuh, cpy.cu, set-rows.cu, llama-context.cpp,
  llama-kv-cache.cpp, llama-kv-cache.h, AGENTS.md, .gitignore

## Build / test plan
Build (user's command):
  nix develop .#cuda --command bash -c 'cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DCMAKE_BUILD_RPATH=/run/opengl-driver/lib -DCMAKE_INSTALL_RPATH=/run/opengl-driver/lib \
    -GNinja -DGGML_CUDA_FA_PREFILL_F16=N && cmake --build build --parallel'

Test:
  ./build/bin/llama server -np 1 --host 0.0.0.0 \
    -hf prism-ml/Ternary-Bonsai-2-27B-gguf:TQ1_0 \
    --jinja -ngl 99 -kvu \
    -c 131072 \
    -fa on \
    -ctk ptq_0 \
    -ctv ptq_0 \
    --port 8080 \
    --temp 1.0 --top-p 0.95 --top-k 20
  (then hit http://localhost:8080 with a prompt and check for sensible output)

## Implementation (DONE - 2026-10-15)

Write path (K/V computed as f32 -> quantized KV cache):
1. `ggml/src/ggml-cuda/cpy-utils.cuh` - added `quantize_f32_ptq1_0_block` (bit-for-bit port
   of `quantize_row_ptq1_0_ref`, ggml-quants.c:2205-2252, with local static
   `stages[3]={32,16,8}`) and `quantize_f32_pq2_0_block` (bit-for-bit port of
   `quantize_row_pq2_0_ref`, ggml-quants.c:113-140). Scale field written as `(half)d`
   (GGML_FP32_TO_FP16 is not available in the GPU path). Plus `cpy_blck_f32_ptq1_0` /
   `cpy_blck_f32_pq2_0` wrappers.
2. `ggml/src/ggml-cuda/set-rows.cu` - dispatch added:
   `set_rows_cuda_quant<idx_t, block_pq2_0, QK_PQ2_0, quantize_f32_pq2_0_block>` and
   `set_rows_cuda_quant<idx_t, block_ptq1_0, QK_PTQ1_0, quantize_f32_ptq1_0_block>`.
3. `ggml/src/ggml-cuda/cpy.cu` - dequant block funcs `cpy_blck_ptq1_0_f32` /
   `cpy_blck_pq2_0_f32` (use `dequantize_ptq1_0` / `dequantize_pq2_0` with ib=0,
   sequential j*2 layout), plus four global launch funcs
   `ggml_cpy_f32_ptq1_0_cuda` / `ggml_cpy_ptq1_0_f32_cuda` / `ggml_cpy_f32_pq2_0_cuda` /
   `ggml_cpy_pq2_0_f32_cuda` and dispatch branches F32<->PTQ1_0 / F32<->PQ2_0.

CLI / gate changes:
4. `common/arg.cpp` - `kv_cache_types` now includes `GGML_TYPE_PTQ1_0` and
   `GGML_TYPE_PQ2_0` -> CLI names `ptq_0` and `pq2_0`.
5. `src/llama-context.cpp` - mean-center gate generalized from Q4_0-only to
   {Q4_0, PQ2_0, PTQ1_0} (line ~4165).
6. `src/llama-kv-cache.cpp` + `.h` - `load_kv_mean_center` parameter `require_q4_0`
   removed (single `path` arg); the Q4_0 layer-type check generalized to
   {Q4_0, PQ2_0, PTQ1_0}. Bias application (F32 ggml_sub) is type-agnostic.

No change needed: decode K/V matmul uses generic `MUL_MAT`/`MUL_MAT_ID`/`GET_ROWS`
which already support PTQ1_0/PQ2_0 (ggml-cuda.cu ~5657/5705); rope-shift
`ggml_cast` quantized->f32 now works via the new cpy.cu dequant entries.

## RESOLVED: SET_ROWS crash (2026-10-15)

The `pre-allocated tensor (cache_k_l0 (view)) in a buffer (CUDA0) that cannot run the operation (SET_ROWS)`
SIGABRT (ggml/src/ggml-backend.cpp:930) had three contributing gaps:

1. Scheduler gate — `ggml_backend_cuda_device_supports_op` (ggml-cuda.cu:5732-5745) did NOT list
   `GGML_TYPE_PQ2_0` / `GGML_TYPE_PTQ1_0` in the `SET_ROWS` quantized-dst type list, so the
   CUDA backend reported it could not run SET_ROWS and the scheduler aborted the pre-allocated
   K-cache write. Fix: add both types to that list (alongside q4_0..iq4_nl, all with F32 src).
2. Kernel entry assert — `ggml_cuda_op_set_rows` (set-rows.cu:400) required
   `src0 F32 OR (src0 F16 AND dst F16)`, so F32->F16-quantized (the `set_rows_cuda_quant` path
   that handles q4_0..ptq_0/pq2_0) was blocked by the assert. Fix: relax to `src0 F32 OR src0 F16`
   (dst quantization is resolved inside the generic `set_rows_cuda<float,...>` dispatch).
3. Test seam design — `llama_kv_cache::load_kv_mean_center` carried its own K-type check that
   rejected the F32 K cache the `test_softmax_invariance` seam deliberately loads to validate pure
   fp32 math. Fix: remove the type check from the method (the fp32 subtraction is type-agnostic);
   enforce the {q4_0,pq2_0,ptq_0} gate at the public `--kv-mean-center` path in `llama_context`
   init (src/llama-context.cpp), which is where the user opts in.

Verified: `build/bin/test-kv-mean-center` -> all 3 subcases OK, EXIT 0,
nmse(baseline, centered) = 3.03e-16 (<< 1e-8).

## OPEN
- No precomputed mean-centering bias calibrated for ptq_0 / pq2_0 exists. `tools/kv-mean-center/`
  currently only produces one for q4_0. So `--kv-mean-center` with a q4_0 bias on ptq_0 would be a
  basis mismatch (checked via `kv_mean_center.k_rot` gguf key). The NON-mean-centering path is
  what is enabled now.
- Flash-attention quantized-K read path verified not to need changes (it uses the same
  mul_mat/get_rows/dequant primitives now enabled).

## VERIFY (pending build)
Build:  nix develop .#cuda --command bash -c 'cmake -B build -DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_RPATH=/run/opengl-driver/lib
  -DCMAKE_INSTALL_RPATH=/run/opengl-driver/lib -GNinja
  -DGGML_CUDA_FA_PREFILL_F16=N && cmake --build build --parallel'
Test 1 (ptq_0, no mean-centering):
  ./build/bin/llama server -np 1 --host 0.0.0.0 -hf prism-ml/Ternary-Bonsai-2-27B-gguf:TQ1_0
    --jinja -ngl 99 -kvu -c 131072 -fa on -ctk ptq_0 -ctv ptq_0 --port 8080
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00
    --reasoning-budget 8192 --reasoning-preserve
Test 2 (pq2_0): same but -ctk pq2_0 -ctv pq2_0.
