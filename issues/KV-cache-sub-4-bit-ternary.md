# Sub-4-bit ternary KV cache on CUDA

Status: **PQ2_0 CUDA smoke test passes; PTQ1_0 read path remains unimplemented** (updated 2026-10-16).

Work started at [`28ffe0d36`](../commit/28ffe0d3661d6785c48a2301e617bd01435ea97e),
which added CUDA cache writes and exposed both cache types in the command-line
options. Follow-up commits on this branch fixed `SET_ROWS` handling and added
native CUDA FlashAttention reads for `PQ2_0`:

- `8525ab8e9` — fix `SET_ROWS` pre-allocation handling and generalize the
  mean-center gate.
- `3a2be8b9a` — add the `PQ2_0` in-place CUDA FlashAttention path.

These commits are ancestors of the current branch head. The `PQ2_0` path has
now passed a CUDA build and a 9B server smoke test. `PTQ1_0` attention reads
remain unimplemented.

## Goal

Use the Prism `PTQ1_0` (group-128 ternary, 1.75 bits per weight) and `PQ2_0`
(group-128 2-bit) formats for both K and V caches on CUDA in `llama-cli` and
`llama-server`, with FlashAttention enabled. The goal is lower KV memory use
while preserving model quality and useful decode performance.

## Assessment

This is achievable in principle. The formats, CUDA quantize/write operations,
and CUDA weight operations already exist. `PQ2_0` also now has an in-place CUDA
FlashAttention implementation in the source tree. `PTQ1_0` does not yet have
that read path, so CUDA FlashAttention cannot consume that cache type directly.

The current NixOS build compiles the `pq2_0-pq2_0` pair, and the 9B smoke test
confirms the target can serve requests with that cache. The evidence is limited
to the tested model shape and short requests; `ptq_0-ptq_0` is not configured or
implemented.

## Current findings

- CUDA K/V writes for `PTQ1_0` and `PQ2_0` are implemented through `SET_ROWS`
  and `CPY` paths. The issue history reports these write paths working.
- The `PQ2_0` FlashAttention path is present in `ggml/src/ggml-cuda/fattn.cu`
  and `fattn-mma-f16.cuh`: the type is admitted by the gate, marked native,
  decoded by a tile loader, dispatched for 128- and 256-wide heads, and has
  kernel instances. `build.sh` also enables the `pq2_0-pq2_0` pair.
- The first `bash build.sh` attempt failed because the generated PQ2_0 vector
  FlashAttention instance instantiated unsupported K and V decoders. The
  build path now omits vector instances for PQ2_0, and native-capable cache
  shapes bypass vector-kernel selection so decode uses the in-place MMA path.
- Fixed-seed comparison against `build.old` with Q8_0 K/V exposed two PQ2_0
  correctness bugs: the first packed block in the native tile loader was read
  with a two-byte offset, and the CUDA cache writer ORed quantized codes into
  uncleared bytes. Both were fixed. After rebuilding, the same 40-token prompt
  produced the same coherent explanation prefix as the Q8_0 baseline; a
  256-token request remained coherent and reported 88.16 generated tokens/s.
  A concurrent `nvidia-smi` sample reached 75% GPU utilization with 5,123 MiB
  allocated. This is a smoke result, not a broad correctness or quality test.
- The current NixOS build succeeded via `bash build.sh` for CUDA architecture
  75, including all 547 targets on the initial build and the subsequent CUDA
  rebuild and links. CMake reported the `pq2_0-pq2_0` FlashAttention pair.
- The native CUDA FlashAttention implementation only covers the instantiated
  head sizes and tilings; it is not generic support for every CUDA device,
  head shape, or mixed K/V type. The target Bonsai model shape must be checked.
- `PTQ1_0` still lacks a native FlashAttention loader and kernel instances.
  Its group-128 base-3 packed representation differs from `PQ2_0`, so its bytes
  must be decoded with a dedicated loader. Its 128-element block size alone
  does not make the PQ2 loader reusable.
- `--kv-mean-center` has no calibration file for either sub-4-bit format.
  Do not apply a Q4_0-calibrated bias to these caches; treat mean-centering as
  outside the initial acceptance criteria until compatible biases are produced
  and evaluated.
- The earlier report of CPU attention fallback for both formats predates the
  current `PQ2_0` implementation. Retain it as historical observation only;
  recheck current runtime logs and device utilization.

## Blockers and risks

1. **PTQ1_0 read path:** implement and instantiate the CUDA FlashAttention tile
   loader, add deliberate type dispatch, and register `ptq_0-ptq_0` in the
   configured FlashAttention quant pairs. Ensure unsupported or mixed type
   combinations are rejected explicitly rather than reaching an unrelated
   quantized loader.
2. **PQ2_0 end-to-end verification:** build the current source and run a real
   Bonsai decode with `-ctk pq2_0 -ctv pq2_0 -fa on`. Confirm the CUDA backend
   executes attention, outputs are sensible, and compare throughput and memory
   use against the Q4_0 baseline.
3. **PTQ1_0 end-to-end verification:** after implementing the read path, run
   the same checks for `ptq_0`; exercise cache writes as well as attention
   reads, including prompt processing and decode.
4. **Quality and performance acceptance:** GPU utilization alone is not a
   correctness or performance test. Record tokens/s, peak cache memory, and a
   repeatable quality comparison against the same model with a supported cache
   type. Define acceptable quality and speed thresholds before calling this
   complete.
## Work plan

1. **Done:** build with `bash build.sh` in the NixOS CUDA shell and confirm the
   `pq2_0-pq2_0` FlashAttention pair compiles for architecture 75.
2. **Initial smoke test done:** run the Q8_0 baseline and PQ2_0 server with the
   same prompt, seed, and model. Broader quality/performance comparison and
   target 27B validation remain. If future runs show attention off the GPU,
   capture the backend support/fallback reason before changing code.
3. Implement the `PTQ1_0` loader and explicitly scoped dispatch/instances.
   Add focused tests comparing its tile output with canonical CPU dequantization
   across representative blocks and boundary layouts.
4. Add `ptq_0-ptq_0` to the configured CUDA FlashAttention quant pairs, rebuild,
   and run the corresponding server smoke test.
5. Measure both formats against the Q4_0 baseline for cache memory, decode
   throughput, and model quality at the same context and prompt settings.

## Commands

Build in the NixOS CUDA development shell:

```sh
bash build.sh
```

Existing mean-center unit test, reported passing previously on CUDA 0:

```sh
./build/bin/test-kv-mean-center
```

The user supplied these commands for the 9B runtime comparison. The first
launch is the existing Q8_0 KV-cache baseline:

```sh
./build.old/bin/llama serve -np 1 --host 0.0.0.0 \
  -hf prism-ml/Ternary-Bonsai-2-9B-gguf:PQ2_0 \
  --jinja -ngl 99 -c 262144 -fa on -ctk q8_0 -ctv q8_0 --port 8080 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --presence-penalty 1.0 --repeat-penalty 1.0 --repeat-last-n 64
```

After the new build succeeds, use its executable with PQ2_0 K/V cache:

```sh
./build/bin/llama serve -np 1 --host 0.0.0.0 \
  -hf prism-ml/Ternary-Bonsai-2-9B-gguf:PQ2_0 \
  --jinja -ngl 99 -c 262144 -fa on -ctk pq2_0 -ctv pq2_0 --port 8080 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --presence-penalty 1.0 --repeat-penalty 1.0 --repeat-last-n 64
```

The local endpoint can be checked with:

```sh
curl -sS --max-time 60 http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain 1-bit ternary quantization in one plain sentence.","max_tokens":32}'
```

The initial pre-fix endpoint check returned 79.16 generated tokens/s, but the
output was not useful and exposed the cache correctness issue. After the
fixes, the rebuilt server returned 88.16 generated tokens/s for 256 generated
tokens and sampled up to 75% GPU utilization. See findings above for the
fixed-seed Q8_0 comparison and the remaining limits of this smoke test.

## Open questions

- Does the Bonsai model's actual attention head shape match a compiled native
  `PQ2_0` FlashAttention instance on the RTX 2080 Ti?
- Are K and V guaranteed to use the same type in the target configuration? If
  mixed types can be requested, is each combination handled or rejected
  explicitly?
- What quality metric and regression threshold define success for each cache
  format?
- Should calibration and quality evaluation for mean-centering be a follow-up
  issue, or part of this feature's acceptance criteria?
