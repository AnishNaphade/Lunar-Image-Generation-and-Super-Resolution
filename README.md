# Lunar Image Generation and Super Resolution using Custom GANs

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Publication](https://img.shields.io/badge/Publication-Scopus%20Elsevier%20ACT--2026-blue?style=for-the-badge)](https://elsevier.com)

An end-to-end deep learning framework integrating **Deep Convolutional Generative Adversarial Networks (DCGAN)** for structural lunar surface synthesis with an **Enhanced Super-Resolution GAN (ESRGAN)** for $4\times$ spatial detail restoration on planetary orbital imagery.

---

## 🌟 Key Highlights

- **Dual-Stage Generative Pipeline:** Synthesizes realistic 64×64 lunar terrain textures from random latent noise ($z \sim \mathcal{N}(0, I)$) and upscales them to high-resolution (256×256+) photorealistic visuals.
- **Architectural Design:** Combines a convolutional generator-discriminator architecture with deep Residual-in-Residual Dense Blocks (RRDBNet) and relativistic adversarial loss.
- **Quantitative Benchmark:** Outperforms standard interpolation baselines across all standard image reconstruction metrics:
  - **PSNR:** **29.92 dB** (+18.2% gain over bicubic interpolation at 25.31 dB)
  - **SSIM:** **0.8634** (+33.4% structural fidelity gain over bicubic at 0.6473)
  - **FID Score:** **0.70**, demonstrating high realism and fidelity to real lunar topography
- **Interactive Visualizer:** Built-in Streamlit explorer dashboard to visualize generated lunar scenes across training epochs.

---

## 🏗️ Architecture Overview

```
                      Stage 1: DCGAN                          Stage 2: ESRGAN (RRDBNet)
  Latent Noise     +--------------------+   Low-Res Surface    +-----------------------+   High-Res Surface
  [z ~ N(0, I)] -> | Transposed Conv2D  | ->  [64 x 64 px]  -> | Residual-in-Residual  | ->  [256 x 256 px]
     (100-D)       | BatchNorm + ReLU   |                      | Dense Blocks (RRDB)   |    (Photorealistic)
                   +--------------------+                      +-----------------------+
```

### 1. DCGAN (Image Synthesis)
- **Generator:** Uses 2D transposed convolutions with batch normalization and ReLU activations to map a 100-dimensional latent noise vector into a 64×64 terrain representation.
- **Discriminator:** Deep convolutional classifier using LeakyReLU ($\alpha = 0.2$) downsampling to distinguish generated topography from genuine orbital captures.

### 2. ESRGAN (Super-Resolution Enhancement)
- Replaces standard residual blocks with **Residual-in-Residual Dense Blocks (RRDB)** without batch normalization to prevent artifact formation.
- Optimized using a compound perceptual loss function incorporating VGG-19 high-level feature representations, relativistic adversarial loss, and $L_1$ pixel-wise loss.

---

## 📊 Experimental Results

| Method | PSNR (dB) ↑ | SSIM ↑ | FID ↓ |
| :--- | :---: | :---: | :---: |
| Bicubic Interpolation Baseline | 25.31 dB | 0.6473 | — |
| **LunaGAN (Ours: DCGAN + ESRGAN)** | **29.92 dB** | **0.8634** | **0.70** |

---

## 📂 Repository Structure

```
.
├── EDI- G A N/
│   ├── DCGAN/
│   │   ├── model.py              # DCGAN Generator & Discriminator network definitions
│   │   ├── train.py              # Training loop with loss computation & TensorBoard logging
│   │   ├── mainGAN.py            # Streamlit interactive visual explorer application
│   │   ├── extract_imgs.py       # Dataset pre-processing and feature extraction
│   │   └── generated_images/     # Output lunar terrain samples synthesized across epochs
│   └── Simple_GAN/
│       └── simpleGAN.py          # Baseline GAN implementation for benchmarking
├── EDI-SRGAN,ESRGAN/
│   ├── Copy of GAN-pytorch.ipynb # ESRGAN training, RRDBNet implementation & evaluation
│   └── lunar/                    # Evaluation sets and lunar crater coordinate references
├── requirements.txt              # Environment dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/AnishNaphade/Lunar-Image-Generation-and-Super-Resolution.git
cd Lunar-Image-Generation-and-Super-Resolution
pip install -r requirements.txt
```

### 2. Launch the Streamlit Interactive Explorer
Launch the web interface to view synthesized lunar terrain across epochs:
```bash
cd "EDI- G A N/DCGAN"
streamlit run mainGAN.py
```

### 3. Training the DCGAN
```bash
python train.py
```

---

## 📜 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@inproceedings{naphade2026lunar,
  title={Image Generation and Super Resolution using Custom GAN's},
  author={Kulkarni, Milind and Naphade, Anish and Jagtap, Alankar and Tamboli, Arman and Ambilwade, Aniket and Wagh, Ashlesha},
  booktitle={Proceedings of Elsevier / Scopus ACT-2026},
  year={2026}
}
```

---

## 👨‍💻 Author
**Anish Naphade**  
B.Tech in Artificial Intelligence & Data Science, Vishwakarma Institute of Technology, Pune  
- [GitHub](https://github.com/AnishNaphade)
- [LinkedIn](https://linkedin.com/in/anishnaphade)
- Email: anishnaphade@gmail.com
