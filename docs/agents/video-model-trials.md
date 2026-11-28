# Downstream video-model trials

This note is for experiments after the first-round scene and physics work. The
first-round acceptance path remains the one in `experiments/skeleton/README.md`.

## Interface boundary

`GLOSSARY.md` and ADR-0006 define the first-round proxy signals as depth and
instance ID, with no RGB. The MP4 palettes in the skeleton are previews of
measurements, not the clean RGB reference video expected by EdenLive or
Matrix-Game 4.0 (MG4). ADR-0009 selects rasterization as the first-round
reference renderer, and ADR-0011 fixes the camera to the input image camera.
Producing an RGB proxy and choosing its appearance are downstream decisions;
neither was settled by the model smoke tests below.

## Verified A800 trials (2026-11-26)

| Model | Weights and test | Result | Limit |
| --- | --- | --- | --- |
| EdenLive 5B few-step | 21 files, 36,730,988,499 bytes; official `cryo-crossing` sample, shortened to 13 frames | 1280×704, 24 fps video decoded to 13 frames | Offline four-step path only; no continuous or real-time measurement |
| MG4 Racing 5B | 16 files, 24,228,472,549 bytes in the official Racing + shared manifest; synthetic RGB input, 5 frames | Native two-step Wan VAE path returned exit code 0; 864×480, 24 fps video decoded to 5 frames | Input was synthetic; no quality, adherence, or real-time claim |

Both file sets passed byte-length and SHA-256 verification. The MG4 VAE is a
hardlink to EdenLive's already verified Wan2.2 VAE because both manifests give
SHA-256 `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`.
The logical MG4 payload is 24.2 GB; `du` reports less additional storage
because of that hardlink.

On `a800`, under the assigned directory in `AGENTS.md`:

- EdenLive weights and download log: `models/EdenLive/` and
  `models/EdenLive/download.log`. The output is at
  `edenlive-trial/output/smoke-13f/save/inference/base/cryo-crossing.mp4`.
- MG4 weights and download log: `models/Matrix-Game-4.0-racing/` and
  `models/Matrix-Game-4.0-racing/download.log`. The output and configuration
  are `mg4-trial/output/quality-5f-py310-torch26/{rendered.mp4,render.json}`.

The EdenLive source was fixed at upstream commit `2ea01626e6eaa32592a3b466d99d9739ceca35b0`;
the MG4 source was fixed at `51913b2bfa78a6d54de912b8348b1d318f4d36e2`.
Official sources: [EdenLive code](https://github.com/VAST-AI-Research/EdenLive),
[EdenLive weights](https://huggingface.co/VAST-AI/EdenLive),
[MG4 code](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-4),
[MG4 release manifest](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-4/releases/models.json).

## Download and environment lessons

1. Check A800 GPU jobs, assigned-directory size, and shared free space before a
   substantial transfer or run. A free GPU is not an allocation; follow the
   host scheduling rules in `asus-host-context`.
2. For these releases, A800-to-ModelScope direct downloads were usable at
   roughly 5–6 MB/s per large file. The EdenLive mirror is
   `VAST-AI-Research/EdenLive`. MG4 Racing is `Skywork/Matrix-Game-4.0`;
   its shared Wan files are mirrored under `PAI/Wan2.2-Fun-5B-Control`.
   Local-Mac Hugging Face download and Mac-to-A800 `scp` were slower in this
   session. Recheck reachability and rate rather than assuming a permanent
   advantage for any host or mirror.
3. Use a pinned upstream file list. MG4's `releases/models.json` has paths,
   sizes, and SHA-256 values; its pinned Hub revision is
   `2c53100457da7abcf0e966f75637c644d3423a63`. EdenLive's selected 5B
   few-step files were checked against its official Hub revision
   `4efc2c4c33af7863f6048e36c2bceb2c30bfae97`. A ModelScope `master`
   URL is mutable: download to `.part`, resume with `curl -C -`, verify size
   and SHA-256, then rename into place. The runtime download logs list each
   `VERIFIED` file and end with `DOWNLOAD_COMPLETE`.
4. Check whether an already verified shared component has the same SHA-256
   before downloading another copy. The Wan2.2 VAE was byte-identical across
   these two trials.
5. Test the smallest complete inference path before fetching a new CUDA stack.
   A800 has SM80 A800 GPUs, 80 GB each, driver 535.129.03, and working
   PyTorch 2.6.0+cu126. EdenLive's offline few-step path used that stack with
   diffusers 0.39.0 and transformers 5.12.1. MG4's native two-step path also
   ran with it: Python 3.10 parsed the source, the existing EdenLive virtual
   environment was used read-only, and MG4-only `einops` and `sentencepiece`
   were placed in `mg4-trial/vendor`. No other Python environment was edited.
   An attempted PyTorch 2.11 wheel transfer was stopped after this test passed.
6. These are compatibility trials outside the vendors' verified stacks.
   EdenLive's real-time service and MG4's documented A100 deployment use
   CUDA 13 family runtimes; NVIDIA lists driver 580 as the minimum for CUDA
   13.x. The A800 driver is 535. Keep real-time performance and long-session
   stability unverified until measured on a supported or deliberately tested
   runtime. See [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html).

## Next comparison gate

Once the project can render a clean RGB proxy from the reconstructed scene,
use the same camera, motion, and occlusion sequence for both models. Record the
exact proxy frames, reference first frame, prompt, model revision, and output
video. Compare vehicle silhouette and trajectory, occlusion order, temporal
continuity, and whether scene appearance follows the reference. Run a longer
clip and a separate throughput test before choosing a model. MG4 Racing is
scene-tuned; the five-frame synthetic smoke test says nothing about how it
generalizes to the project's reconstructed scene.
