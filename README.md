# PyTorch 1-Hour Intro Workshop

A hands-on 60-minute PyTorch introduction for researchers, built for the MIDAS
Summer Academy. Covers tensors, autograd, the training loop, Dataset/DataLoader,
and an AI-assisted coding demo (Generate → Audit → Modify with Claude).

## Quick Start

Click the badge at the top of `pytorch_intro_workshop.ipynb` to open it directly
in Google Colab — no setup required, runs on Colab's free T4 GPU.

Or open manually:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xiaosuhu/Pytorch_1hour_intro/blob/main/pytorch_intro_workshop.ipynb)

## Repo Structure

```
pytorch_intro_workshop.ipynb        # the workshop notebook
data/
├── pytorch_demo_spectrograms.npz   # small synthetic dataset (see note below)
└── generate_synthetic_data.py      # script used to generate it
```

## About the Data

`pytorch_demo_spectrograms.npz` is **synthetic** — generated to loosely resemble
2D EEG spectrogram data (4-channel, 6-class) purely to give the training loop
something realistic-shaped to chew on. It is **not** real EEG or medical data,
and has no connection to any research dataset. See `generate_synthetic_data.py`
for exactly how it was made.

## Workshop Outline (60 min)

1. Why PyTorch? — mental model shift from sklearn
2. Core Three — Tensor, Autograd, `loss.backward()`
3. Training Loop Anatomy — every line, every reason
4. AI-Assisted Coding Demo — Generate → Audit → Modify
5. Top 5 Bugs beginners hit
6. Next steps

## Author

Xiao-Su Hu (Frank) — Data Scientist, MIDAS, University of Michigan
