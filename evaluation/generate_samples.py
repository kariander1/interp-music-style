import os
import re
import torch
from pathlib import Path
from scripts.txt2img import txt2img, load_model
import itertools
import random


# Set paths
N_STYLE_PAIRS_PER_CONTENT = 20
ckpt_path="models/ldm/sd/model.ckpt"
content_root = "data/musicTI_dataset/images/content"
style_root = "data/musicTI_dataset/images/timbre"
logs_root = "logs"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(ckpt_path, device)

# Get content and style directories
content_dirs = [d for d in os.listdir(content_root) if os.path.isdir(os.path.join(content_root, d))]
style_dirs = [d for d in os.listdir(style_root) if os.path.isdir(os.path.join(style_root, d))]

# Function to get latest embedding path for a style
def get_latest_embedding_path(style_name):
    matching_logs = [
        d for d in os.listdir(logs_root)
        if os.path.isdir(os.path.join(logs_root, d))
        and d.startswith(style_name)
        and d.endswith(style_name)
    ]

    if not matching_logs:
        return None

    def extract_timestamp(name):
        match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", name)
        return match.group(1) if match else ""

    matching_logs.sort(key=lambda x: extract_timestamp(x), reverse=True)
    latest_log = matching_logs[0]
    path = Path(logs_root) / latest_log / "checkpoints" / "embeddings.pt"
    return path if path.exists() else None

# All unordered style pairs (no repetition)
style_pairs = list(itertools.combinations(style_dirs, 2))

# Loop over all content dirs and style pairs
for content in content_dirs:
    content_path = os.path.join(content_root, content)
    print(f"\n🔍 Content: {content_path}")

    random_sampled_styles = random.sample(style_pairs, k=N_STYLE_PAIRS_PER_CONTENT)
    for style1, style2 in random_sampled_styles:
        emb1 = get_latest_embedding_path(style1)
        emb2 = get_latest_embedding_path(style2)

        if emb1 and emb2:
            print(f"✅ ({style1}, {style2})\n   → {emb1}\n   → {emb2}")
            txt2img(
                outdir=f"outputs/{content}/{style1}_{style2}",
                ddim_steps=50,
                ddim_eta=0.0,
                n_iter=2,
                n_samples=1,
                scale=5.0,
                model=model,
                embedding_path=[emb1, emb2],
                alpha=[0.5, 0.5],
                content_path=content_path,
                strength=0.7,
                first_content_only=True,
                convert_to_audio=True,
            )
        else:
            print(f"⚠️  Skipping pair ({style1}, {style2}) due to missing embedding(s).")


    # === 2. Run on single (content, style) pairs ===
    print(f"\n🎯 Running single-style generations for {content}...")
    for style in style_dirs:
        emb = get_latest_embedding_path(style)
        if emb:
            print(f"🎵 {content} x {style} → {emb}")
            txt2img(
                outdir=f"outputs/{content}/{style}",
                ddim_steps=50,
                ddim_eta=0.0,
                n_iter=2,
                n_samples=1,
                scale=5.0,
                model=model,
                embedding_path=[emb],
                alpha=[1.0],
                content_path=content_path,
                strength=0.7,
                first_content_only=True
            )
        else:
            print(f"⚠️  Skipping style {style} (no embedding found).")