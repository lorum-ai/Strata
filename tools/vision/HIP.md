# Experimental HIP image encoder

This builds Strata's existing mtmd encoder with llama.cpp's HIP backend. It uses the same model's mmproj file
and SVE1 embeddings as the CUDA and CPU builds.

## Install on Windows

Run `START-HERE.bat --vision gpu` (or `--vision yes`) to opt in. Images stay off when no vision option is supplied.
Visual Studio C++ Build Tools with the Desktop development with C++ workload are required. Setup installs a private
ROCm SDK in `.rocm-win`, builds the encoder for the selected card, downloads the matching model family's mmproj
through the existing resumable downloader, and writes the vision configuration. The first SDK download takes
several GB. `STRATA_ROCM_VERSION`, `STRATA_ROCM_INDEX` and `ROCM_VENV` override the SDK defaults, as in the HIP engine
build script. The default SDK is ROCm `10.2.0a20260930`.

The encoder is installed in `engine/vision-hip`. Its runtime dependency DLLs are copied beside it and its own
library path is supplied to the encoder process, without replacing the language engine's DLLs or library path.
Setup records the backend, source, llama.cpp revision, architectures and SDK in `engine/BUILD.json`; changed or
missing encoders are rebuilt or redeployed on the next setup run. GPU mode requires a visible HIP device and a
successful warm-up before `READY`. The existing server starts the encoder before sizing the language engine cache.

Linux setup's image choices remain unchanged; Windows CPU vision is still outside this integration.

## Manual build

On Windows, use Visual Studio C++ Build Tools, CMake, Ninja and the private ROCm SDK used by
`tools/hip/build_windows.bat`. Initialize that SDK with `.rocm-win\Scripts\rocm-sdk.exe init`.
The llama.cpp checkout in `third_party/llama.cpp` must match `LLAMA_CPP_COMMIT` in `setup.py`.

Run from the repository root:

```bat
tools\vision\build_hip_windows.bat
```

The script defaults to gfx1100 (RX 7900 XT / XTX). Set `STRATA_HIP_ARCHS` to change it. Build output is in
`build-vision-hip\bin`; the script copies the SDK's HIP runtime and compiler-support DLL beside the encoder.
When running the encoder, also put the SDK's `bin` directory on PATH for hipBLAS and rocBLAS.

For a standalone image test, run `strata-vision.exe --mmproj <projector.gguf> --model <first-model-shard.gguf>
--gpu --min-tokens 1024 --max-tokens 1024`. After `READY`, send `ENC <image-path> <output-path>` on stdin, then
`QUIT`. Use paths without spaces for this diagnostic. `OK` reports the tokens, grid dimensions and encode time.
The output must contain a 20-byte SVE1 header and `tokens * embedding_width` float32 values.

Before integration, compare those values against a CPU run using the same image, weights, token limits and
flash-attention setting. Check finite values, grid dimensions and numerical differences, then test image answers
through the API. A successful compile alone does not validate GPU encoding.

Use a separate server config while testing. Its `vision` section names the encoder, projector and first model
shard, with `gpu: true`. Its engine arguments need `--vision`. Start the encoder before the language engine so
its warm-up allocations are included when the engine sizes the expert cache. Keep the test server bound to
127.0.0.1. Measure VRAM with both processes loaded before selecting a reserve or a larger image-token limit.

## Local validation, 2026-10-10

Windows, RX 7900 XTX (gfx1100, 24 GB), 64 GB system RAM, ROCm 10.2.0a20260930, pinned llama.cpp
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, BF16 projector, and the prebuilt Strata 0.1.41 HIP language engine
with IQ3_XXS. These are smoke tests, not a quality benchmark.

- CPU and HIP builds completed. The CUDA/HIP mutually exclusive option check rejected both being enabled.
  CUDA and SYCL builds were not tested on this machine.
- A generated 896 x 896 image containing a red rectangle and blue circle encoded to 1,024 tokens, a 32 x 32
  grid, and 2,560 floats per token. All CPU and HIP output values were finite.
- With flash attention explicitly off on both encoders, HIP reported 331 ms for that image. Startup including
  warm-up took 2.85 s. The CPU run reported 29,149 ms while the HIP build was active, so this is not a controlled
  CPU/GPU speed comparison.
- CPU versus HIP embeddings: cosine similarity 0.998639774, relative RMS error 0.052141831, maximum absolute
  difference 0.08243154. Outputs are not bit-identical. These numbers and the image-answer checks below do not
  establish quality parity across arbitrary images.
- With the server's default attention setting, Windows GPU Process Memory counters reported 1,530,564,608 bytes
  (1.43 GiB) of dedicated GPU memory for the resident vision encoder after warm-up.
- OpenAI image requests correctly identified a red rectangle on the left / blue circle on the right, then a
  separate green circle on the left / purple rectangle on the right. A subsequent text-only request returned
  the requested literal `READY`. The test config used an 8,192-token context and a 700 MiB engine reserve, with
  the encoder started first. Larger contexts, OCR, photographs, multiple-image requests and Anthropic image
  requests have not been tested.
- Under the agent sandbox, language-model prefill failed with rocBLAS `hipErrorInvalidKernelFile`. The same
  configuration succeeded when launched outside that sandbox. This does not require an elevated Windows
  console for the user's normal launcher.

The measurements above describe the original manual build. They do not validate later installer changes.
Machine-specific launchers and configs, binaries, dependencies and model weights are not part of this contribution.
