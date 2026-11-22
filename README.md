# Aquarelab

A lab that adapts color palettes on any image using convolutional neural networks.

Upload a photo and Aquarelab identifies its main tones, separates each region by what it represents and shows the resulting palette. Then upload a new palette with the same number of colors and get the photo recolored, with the original's light, shadow and texture preserved.

> Status: in development (v1)

## The problem

Traditional color editors group pixels by color similarity alone. If a photo has a blue shirt and a blue sky, both fall into the same palette color, and changing one changes the other too.

Aquarelab groups pixels by what they represent, not only by the color they have. Sky, skin, vegetation and clothing can receive different colors, even when they share the same tone in the original photo.

## How it works

1. The user uploads a color photo.
2. The network identifies the main tones of the image, up to a maximum of 8, and maps which regions belong to each tone.
3. The extracted palette is displayed.
4. The user uploads a new palette with the same number of colors.
5. Each new color is assigned to an original tone, from darkest to brightest, preserving the image's contrast.
6. The photo is recolored instantly.

## The idea behind the model

The network does not paint the image. It understands the image and produces two outputs:

- the main tones of the photo, leaving out the ones that are not relevant;
- a map showing how much each pixel belongs to each tone.

Recoloring is done on top of that map. Instead of replacing each pixel's color with the new one, Aquarelab applies a shift: if a blue tone became red, every pixel in that region receives the same color shift, proportional to how much it belongs to it. This keeps shadows, highlights and textures in place, and the final image looks like the same photo painted with different colors.

This separation brings three advantages:

- Preserving light and shadow is guaranteed by the math, not by the network getting it right.
- Changing the palette is instant, because the network does not need to run again.
- Each new color only changes the regions linked to the tone it replaces.

To separate regions by meaning, and not only by color, the network starts from a pretrained backbone that already recognizes objects and textures.

## Training

Training uses no manual labels. The network learns by trying to rebuild the original photo using only the palette and the map it generated itself. It is rewarded for faithful reconstructions that use few tones and well-defined regions. Any large public collection of color photos works as training data.

## Evaluation

The network is compared against a classic palette extraction method that groups pixels by color alone. The goal is to show, with examples and metrics, in which situations the network separates regions that the classic method mixes, and where it still fails.

Experiment history and analysis: see [notebooks/README.md](notebooks/README.md).

## Roadmap v1

- [x] Recoloring by color shift from a tone map
- [x] Palette assignment by luminance order
- [ ] Network that extracts the main tones and the map of each region
- [ ] Training by reconstruction, without labels
- [ ] Interface to upload the photo and the palette
- [ ] Comparison with the classic method

## Next versions

- A critic network that evaluates how natural a recolored image looks
- Automatic color assignment more sophisticated than luminance order
- Support for black and white photos

## Known limitations

Luminance-based assignment preserves contrast, but does not always produce the combination a person would choose. When the new palette is very different from the original, some regions may receive unnatural colors, such as skin in unlikely tones. The maximum of 8 tones also simplifies very complex images.

## Learning goals

This project was created to deepen knowledge of convolutional networks beyond image classification, exploring networks that produce maps as output, pretrained models as a backbone, training without labels, and evaluation against classic methods.