# E-DRNet — Low-Light Image Enhancement

> Decompose the dark. Recover the detail. Skip the artifacts.

Most low-light enhancers do one of two things: they brighten everything and blow out your highlights, or they denoise so aggressively the image looks like a watercolour painting. E-DRNet tries to do neither. It separates the image into frequency components first, fixes each one independently, then puts it back together. Simple idea. Surprisingly annoying to get right.

Tested across five benchmark datasets (LOL-v1, LOL-v2, LIME, DICM, MEF). Beats RetinexDIP and Zero-DCE on PSNR and SSIM. Runs at ~15ms inference on a GPU, so it's actually usable in real-time pipelines.

---

## Why I made this

Every existing method I tried during my final year project either introduced colour halos, smoothed out textures I wanted to keep, or took so long to run it was useless for anything live. The core problem, I realized, is that most models work in RGB and brightness is tangled up with colour — so when you try to fix one, you mess up the other.

The fix turned out to be annoyingly simple in hindsight: switch to HSV, isolate the Value channel, do all the hard work there, and leave Hue and Saturation completely untouched. Then use a learnable Gaussian filter to split the Value channel into low-frequency (lighting) and high-frequency (texture + noise) components and process each separately. The Frequency Transformer Layer handles the high-frequency side — it uses self-attention to tell the difference between actual edges and random sensor noise, which CNNs are genuinely bad at.

This started as my thesis. I'm putting it here because it shouldn't live only in a PDF.

---

## Getting started

```bash
git clone https://github.com/yourusername/E-DRNet.git
cd E-DRNet
pip install -r requirements.txt
```

To run inference on a single image:

```bash
python inference.py --input your_dark_image.jpg --output enhanced.jpg
```

To train from scratch on LOL-v1:

```bash
python train.py --dataset LOL-v1 --epochs 100 --batch_size 8
```

Trained on Google Colab with a T4 GPU. If you're running locally, anything with 6GB+ VRAM should be fine.

---

## How it works

The pipeline has four stages and they all matter.

**Stage 1 — Frequency Decomposition.** Input RGB gets converted to HSV. The Value channel goes through a learnable Gaussian filter (the variance `σ` is a trainable parameter, not fixed) which splits it into a low-frequency illumination map and a high-frequency detail+noise map. Hue and Saturation are saved untouched — they don't get processed at all.

**Stage 2 — High-Frequency Recovery (HRB).** The noisy high-freq map goes into the Frequency Transformer Layer (FTL). It uses global self-attention (QKV mechanism) across the whole image to figure out what's an edge and what's noise. A spatial skip-connection runs in parallel to make sure fine details aren't accidentally smoothed out during denoising.

**Stage 3 — Illumination Adjustment.** The low-freq map gets adaptive gamma correction. Non-linear, so it lifts dark regions without overexposing areas that are already bright.

**Stage 4 — Reconstruction.** Clean high-freq + corrected low-freq get merged back into the V channel. That gets fused with the original H and S through an Identity Bridge. Pixel shuffling handles upscaling without eating too much compute.

Loss function is a hybrid: L1 + SSIM + perceptual loss. Using all three made a real difference in output quality compared to using any one alone.

---

## Results

| Dataset | PSNR | SSIM |
|---------|------|------|
| LOL-v1  | 25.3 dB | 0.93 |
| LOL-v2  | 24.8 dB | 0.91 |
| LIME    | 23.6 dB | 0.89 |
| DICM    | 22.9 dB | 0.88 |
| MEF     | 23.1 dB | 0.87 |

Compared to Zero-DCE and RetinexDIP on LOL-v1, E-DRNet wins on both metrics and runs faster. Full comparison table is in the thesis (linked below).

---

## Known issues / what's missing

- No pretrained weights uploaded yet — you'll have to train from scratch for now. Working on it.
- The LOL dataset needs to be downloaded separately (it's not mine to redistribute). Instructions are in `data/README.md`.
- Extremely dark images (underexposed to the point of near-black) sometimes produce slightly washed-out results. The gamma correction can be aggressive at the low end.
- No web demo yet. Would like to build one eventually.
- Only tested on still images. Video pipeline is on the list but hasn't happened.

---

## Contributing

If you find a bug, open an issue. If you want to try something — a different loss function, a different attention mechanism, whatever — just fork it and see what happens. PRs are welcome. No formal contribution process, just be reasonable about it.

If you're using this for your own research, a mention in your acknowledgements would be appreciated but isn't required.

---

## Paper / thesis

This repo is based on my undergraduate thesis:
> *A Decomposition-Recomposition Model for Simultaneous Brightness and Noise Optimisation* — [link coming soon]

---

## License

MIT — do what you want, just don't remove the attribution.
