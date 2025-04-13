# 🎵 Style Embedding Interpolation in Music Generation

![Teaser Image](assets/teaser.jpg)

This project extends [MusicTI (AAAI 2024)](https://github.com/lsfhuihuiff/MusicTI_AAAI2024) by introducing *style interpolation* capabilities for music generation. We evaluate how interpolating between different music style embeddings affects the generated audio, exploring smooth transitions and fusion of stylistic elements.

This work was carried out as part of the "Advanced Topics in Audio Processing Using Deep Learning" course at Tel Aviv University.
Audio samples available at https://kariander1.github.io/interp-music-style/, or can be manually accessed via `html.index`.

---

## 🛠️ Installation

**Please use python 3.10.**

```bash
# Clone the repository
git clone https://github.com/kariander1/interp-music-style
cd interp-music-style

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

You will also need the official Riffusion text-to-image checkpoint, available through the [Riffusion project page](https://github.com/riffusion/riffusion). 

Currently, the model can be downloaded by running:

```
mkdir -p models/ldm/sd/
wget -O models/ldm/sd/model.ckpt https://huggingface.co/riffusion/riffusion-model-v1/resolve/main/riffusion-model-v1.ckpt
```
---

## 🏋️ Training / Inversion

This step performs **textual inversion** to learn a new style embedding using mel-spectrogram images.

For example, to learn the embedding of the *accordion* style:

```bash
python main.py --base configs/stable-diffusion/v1-finetune.yaml -t \
               --actual_resume models/ldm/sd/model.ckpt \
               -n test --gpus 0, \
               --data_root images/style/accordion
```

- This command fine-tunes the model using the provided checkpoint and a dataset of mel-spectrograms for a specific style.
- The learned style embedding is saved by default under the `logs` directory.
- Two learnt embedding samples are added for `accordion` and `chime` styles.


## 🎧 Inference

To generate music using a learned style embedding or a combination of styles, run the following commands with the inference script.

---

### 🔹 Example 1: Single Style Generation

Generate audio with a single style (e.g., "accordion"):

```bash
python -m scripts.txt2img \
    --ddim_eta 0.0 \
    --n_samples 1 \
    --n_iter 2 \
    --scale 5.0 \
    --ddim_steps 50 \
    --strength 0.7 \
    --content_path images/content/violin \
    --embedding_path logs/accordion2025-04-04T11-44-43_test/checkpoints/embeddings.pt \
    --alpha 1.0 \
    --ckpt_path models/ldm/sd/model.ckpt \
    --outdir test
```

---

### 🔸 Example 2: Interpolated Style Generation

Interpolate between two styles (e.g., "accordion" and "chime"):

```bash
python -m scripts.txt2img \
    --ddim_eta 0.0 \
    --n_samples 1 \
    --n_iter 2 \
    --scale 5.0 \
    --ddim_steps 50 \
    --strength 0.7 \
    --content_path images/content/violin \
    --embedding_path \
        logs/accordion2025-04-04T11-44-43_test/checkpoints/embeddings.pt \
        logs/chime2025-04-02T12-43-31_test/checkpoints/embeddings.pt \
    --alpha 0.5 0.5 \
    --ckpt_path models/ldm/sd/model.ckpt \
    --outdir test
```

- `--alpha` defines the weight of each style embedding.
- Outputs are saved as audio files and optionally as spectrograms in `outdir`.
- Examples for interpolated samples are available in `assets/samples`.

## 📊 Reproducing Results / Metrics

### Step 1: Download the Official Dataset

Use the following command to download the dataset:
```bash
gdown --output data/musicTI_dataset.zip 1_RjOMwaW8eFrm1dBNtKeuDUAZ9ZG9-Sh2H
```

Or download it manually from:  
https://drive.google.com/file/d/1_RjOMwaW8eFrm1dBNtKeuDUAZ9ZG9-Sh/view

Then unzip the dataset:
```bash
unzip data/musicTI_dataset.zip -d data/musicTI_dataset
```

This will download all required **audio files**.

---

### Step 2: Convert Audios to Images

To convert all audio files into mel-spectrogram images, run:
```bash
python -m data.audios_to_images
```

---

### Step 3: Learn Style Embeddings

Refer to the [Training](#-training--inversion) section to train each style individually.

Alternatively, you can learn embeddings for **all styles** using:
```bash
./evaluation/train_all.sh
```

---

### Step 4: Generate Samples

After training embeddings, generate samples using:
```bash
python -m evaluation.generate_samples
```

This script will use the learned embeddings and content inputs to generate outputs for evaluation, both dual (interpolated) styles and single styles.

---

### Step 5: Run CLAP-based Evaluation

Compute content preservation and style alignment metrics using:
```bash
python -m evaluation.clap_eval
```

- Evaluates cosine similarities between audio and text CLAP embeddings.


## 📝 Notes

- **Riffusion**: Used as the base diffusion model for audio generation. It operates on mel-spectrograms by treating them as images and is built on top of Stable Diffusion.
- **Stable Diffusion**: Serves as the foundational architecture for Riffusion. Originally developed for text-to-image generation.
- **CLIP**: Provides the text encoder used for conditioning the diffusion model on style prompts via learned embeddings.
- **Time-Varying Textual Inversion (TVTI)**: Introduces time-varying embeddings that evolve during the diffusion process, allowing more expressive and temporally-aware style control.
- **CLAP**: Used for evaluation. It maps audio and text to a shared embedding space and is employed to assess both style fit and content preservation.


## 📁 Acknowledgments

Based on [MusicTI (AAAI 2024)](https://github.com/lsfhuihuiff/MusicTI_AAAI2024).  
We extend their work by introducing and evaluating style embedding interpolation for expressive audio synthesis.


