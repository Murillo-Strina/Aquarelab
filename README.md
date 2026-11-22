# Aquarelab

Recolor any photo with a new palette, using a convolutional neural network that understands what each region of the image represents.

Upload a photo and Aquarelab extracts its main tones. Change any tone to recolor only that region, or apply a whole palette at once. Light, shadow and texture are kept.

> Status: in development (v1). The model and the interface work locally; public release and the final evaluation are in progress.

## What it does

- Extracts up to 8 main tones from any photo and shows their hex values.
- Recolors one tone at a time, or a whole palette at once, matched to the original tones from darkest to brightest.
- Keeps the brightness of every pixel, so shadows, highlights, texture and the mood of dark scenes are preserved.
- Outputs the result at the original resolution, up to 2048 px on the longer side, without lossy compression.

## The problem

Traditional color editors group pixels by color similarity alone. If a photo has a blue shirt and a blue sky, both fall into the same palette color, and changing one changes the other too.

Aquarelab groups pixels by what they represent, not only by the color they have. Sky, skin, vegetation and clothing can receive different colors, even when they share a similar tone in the original photo.

## How it works

1. The user uploads a color photo.
2. The network runs once and produces the main tones of the image and a map of how much each pixel belongs to each tone.
3. The extracted tones are shown as editable colors.
4. The user changes one tone, or applies a whole palette, which is matched to the original tones by luminance.
5. The photo is recolored instantly, because recoloring only uses the stored maps and does not run the network again.

## The model

**Architecture.** PaletteNet is a U-Net style network. The encoder is a ResNet-34 pretrained on ImageNet (via `timm`), which already recognizes objects and textures. A decoder written for this project turns its features back into a full-resolution map with 8 channels, one per tone, normalized with a softmax so each pixel's memberships sum to 1.

**Palette.** Each tone's color is the weighted average of the pixels that belong to it. The palette is not predicted directly: it follows from the map.

**Recoloring.** Instead of replacing each pixel's color, Aquarelab shifts it in Lab color space. If a tone changes from blue to red, every pixel receives that shift in proportion to how much it belongs to the tone. Only hue and saturation shift; brightness stays as in the original. This brings three advantages:

- Preserving light, shadow and texture is guaranteed by the math, not by the network getting it right.
- Changing the palette is instant, because the network does not run again.
- Soft memberships make transitions between regions gradual, so gradients such as fog or sky change color smoothly.

**Inference.** The network analyzes a copy of the photo reduced to 512 px, which keeps analysis fast on a CPU. The tone maps are then upscaled back to the original size before recoloring. The exported weights take 98 MB, in the `safetensors` format.

## Training

Training uses no manual labels. The network learns by rebuilding each photo using only its own palette and map. The loss has three parts:

- **Reconstruction:** L1 distance in Lab between the photo and its rebuilt version.
- **Entropy (weight 0.01):** pushes each pixel to commit to fewer tones, while keeping boundaries soft.
- **Usage floor:** penalizes tones used in less than 2% of a batch, so no tone dies. It is averaged over the batch, so simple images may use fewer tones as long as every tone is useful somewhere.

**Data and setup.** COCO train2017 (118k color images, resized to a 320 px short side), random 256 px crops, batch of 32, 10,000 steps, Adam with a cosine learning rate schedule from 1e-3, float16 mixed precision. Training took about one hour on a Kaggle T4.

**Results on 500 unseen images.** Every tone is used across the set, images use 7.6 tones on average, and the reconstruction error is 0.026. A stronger entropy weight (0.05) was also tested: it made memberships nearly hard, which turned tone boundaries into visible color edges, so 0.01 became the reference model.

The full history, from a k-means baseline to the final training runs, is in [notebooks/README.md](notebooks/README.md).

## Observed limitations

- **Palettes that mix opposite hues** can create colored bands in large gradients, such as a night sky, and rings around lights. Neighboring tones receive very different hues, and the change between them becomes visible. Palettes within a single color family avoid this, which is why the built-in palettes follow that rule.
- **Tones that mix different colors** can shift to unexpected hues. When eyes and skin share a brightness range and fall into the same tone, a shift designed for the skin can turn the eyes an unexpected color.
- **Fine textures** such as fur or grass are split by light and dark within the texture, so they can look speckled with two colors.
- **Rare, very saturated colors** can be merged into a neighboring tone, since few training images contain them.
- **Small or heavily compressed images** show their compression artifacts more after recoloring. Larger, cleaner sources give clearly better results.
- **Assignment by luminance** preserves contrast, but does not always produce the combination a person would choose.
- **Training data** is mostly well-lit everyday photos. Night scenes and stylized illustrations work, but were rarely seen during training.
- **A thin colored line** can appear along the bottom edge of some results. It is still being investigated.

## Getting started

**Requirements:** Python 3.11 to 3.14 and [uv](https://docs.astral.sh/uv/). The app runs on CPU. Training is much faster with an NVIDIA GPU.

```bash
git clone https://github.com/Murillo-Strina/Aquarelab.git
cd Aquarelab
uv sync
```

**Model weights.** The trained weights are not published yet. To use your own trained model, export the weights from a training checkpoint:

```bash
uv run python -m aquarelab.inference checkpoints/<run>/step_010000.pt models/palettenet_v1.safetensors
```

**Run the app:**

```bash
uv run --group app python app/app.py
```

Then open `http://127.0.0.1:7860`. A different weights file can be set with the `AQUARELAB_MODEL` environment variable.

**Run the tests:**

```bash
uv run pytest
```

**Train a model.** Download COCO train2017 and val2017 into `datasets/coco/`, resize them once, then train:

```bash
uv run python -m aquarelab.prepare_data datasets/coco/train2017 datasets/coco/train2017_320
uv run python -m aquarelab.train --config configs/base.toml
```

Any value in `configs/base.toml` can be changed from the command line, for example `--set loss.entropy_weight=0.05`. Metrics and preview images are logged to TensorBoard (`uv run tensorboard --logdir runs`).

## Project structure

```
app/                Gradio interface
configs/            Training configuration
notebooks/          Lab journal: experiments, figures and conclusions
src/aquarelab/
  model/            PaletteNet
  losses.py         Reconstruction, entropy and usage floor losses
  data.py           Image loading and batching
  prepare_data.py   One-time resizing of training images
  train.py          Training script
  inference.py      Analysis, recoloring and weight export
  recolor.py        Color conversions and palette assignment
  metrics.py        Tone ownership and usage metrics
  baseline.py       k-means baseline
tests/              Tests for every module
```

## Roadmap v1

- [x] Recoloring by color shift from a tone map
- [x] Palette assignment by luminance order
- [x] Network that extracts the main tones and the map of each region
- [x] Training by reconstruction, without labels
- [x] Interface to upload the photo and edit the palette
- [ ] Public release of the model and the interface on Hugging Face
- [ ] Comparison with the classic method, using COCO segmentation annotations

## Next versions

- Color assignment that considers each tone's original hue and its neighboring tones, not only luminance
- A critic network that evaluates how natural a recolored image looks
- Support for black and white photos

## Learning goals

This project was created to deepen knowledge of convolutional networks beyond image classification, exploring networks that produce maps as output, pretrained models as a backbone, training without labels, and evaluation against classic methods.