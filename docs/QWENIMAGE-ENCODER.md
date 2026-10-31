# Qwen-Image-2.1 VAE encoder

Use `c/qwenimage_encoder.h` to encode one RGB or RGBA image into normalized
VAE latents on the CPU. The encoder uses float32 and shares the decoder's
convolution, residual block, RMS normalization, attention, and GEMM kernels.

Image editing in the CLI or server still needs a vision encoder, reference
token assembly, and DiT conditioning. This change adds the VAE encoder only.

## Interface

```c
QiVaeEncoder *v = qie_load("path/to/vae");
/* Check v, then allocate height/scale * width/scale * z_dim floats. */
int rc = qie_encode(v, pixels, height, width, channels, normalized_latents);
qie_free(v);
```

### Checkpoint and input

`qie_load` reads `config.json` and the `encoder.*` and `quant_conv.*`
safetensors. It skips decoder weights and temporal-downsampler convolution
weights, which the single-frame path does not use. It accepts the residual,
non-patchified Qwen-Image-2.1 architecture and rejects unsupported
configurations or missing required tensors.

Pass tightly packed HWC RGB or RGBA bytes to `qie_encode`. An RGBA encoder
adds opaque alpha to RGB input and preserves supplied alpha without
compositing. An RGB encoder accepts RGB input only. For float input, use
`qie_encode_f32` with HWC values in `[-1, 1]` and the configured channel count.

Height and width must be positive multiples of the configured spatial scale
(16 for Qwen-Image-2.1). Invalid dimensions or channel counts return `-1`
before pixel access. Prepare the dimensions before calling the encoder;
it does not resize or crop images.

### Output

Allocate `(height / scale) * (width / scale) * z_dim` floats. The output uses
HWC order and contains the posterior mode after `quant_conv`, normalized as
`(mode - latents_mean) / latents_std`. Sampling, DiT patch packing, tiling,
and multi-frame caching are outside this interface.

The downsampler matches
[Diffusers AutoencoderKLQwenImage21](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py):
stride-2 convolution uses bottom/right padding, and the AvgDown3D shortcut
counts the zero-padded temporal slot in its divisor, even for a single image.

## Reproducible CPU checks

Run the dependency-free primitive and invalid-input tests with
`make -C c check`, or run them on their own:

```sh
make -C c tests/test_qwenimage_encoder
./c/tests/test_qwenimage_encoder
```

The reference check generates tiny random weights on two CPU threads and
cleans up its temporary fixtures on success or failure. It downloads no model
weights. Install the pinned reference dependencies, including the Diffusers
commit linked above, in an isolated environment:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r c/tools/requirements-qwenimage-tiny.txt
make -C c qwenimage-encoder-check PYTHON=../.venv/bin/python
# Same checks, with ASan/UBSan on the encoder, plus a saved JSON report:
.venv/bin/python c/tools/check_qwenimage_encoder.py --sanitizers --report encoder-check.json
```

The runner sets a timeout for each subprocess, returns a nonzero exit code
on failure, and records commands, output, status, and timing as JSON. It checks:

- Six RGB/RGBA inputs at 16×16, 32×48, 48×32, and 80×112, including the same
  RGB values with opaque and transparent alpha.
- Raw and normalized posterior mode, default and minimal work bands, an
  encoder-only checkpoint, invalid configurations, and a missing intermediate
  weight.

The encoder tolerance is `1e-4` absolute error. The runner also checks the
shared kernels against the existing decoder oracle, with its original
acceptance thresholds. The `qwenimage-tiny-check` CI job includes normal and
ASan/UBSan encoder runs. LeakSanitizer is disabled.

The default CPU build and encoder unit tests need neither Torch nor Diffusers.
Other `make check` tests may use optional dependencies when installed.

These checks cover numerical agreement with tiny random weights. Trained
checkpoints, visual quality, editing instruction-following, non-CPU backends,
large-image peak memory, and downstream reference conditioning remain untested
here. No performance or leak-coverage claim accompanies these checks.
