import os
import torch
import itertools
import torch.nn.functional as F
from transformers import ClapProcessor, ClapModel
import numpy as np
import torchaudio
from tqdm import tqdm

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=-1).item()

def compute_clap_audio_embedding_from_mel(inputs, processor, clap_model, device):
    waveform, sample_rate = torchaudio.load(inputs)
    waveform = waveform.squeeze().numpy()
    inputs = processor(audios=waveform, sampling_rate=48000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = clap_model.get_audio_features(**inputs).float()
    return F.normalize(features, p=2, dim=-1)

def compute_clap_text_embedding(text, processor, clap_model, device):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = clap_model.get_text_features(**inputs)
    return F.normalize(features, p=2, dim=-1)

def main():
    # === Config ===
    output_root = "outputs"
    data_root = "data/musicTI_dataset"
    style_root = os.path.join(data_root, "audios", "timbre")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Load models ===
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    clap_model.eval()

    # === Directories ===
    content_dirs = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
    style_dirs = [d for d in os.listdir(style_root) if os.path.isdir(os.path.join(style_root, d))]
    style_pairs = list(itertools.combinations(style_dirs, 2))

    content_preservations_singles = []
    style_fits_singles = []
    content_preservations = []
    style_fits = []
    single_style_fits = []

    style_fit_singles = {}

    # === Main Evaluation Loop ===
    for content in tqdm(content_dirs, desc="Contents"):
        content_path = os.path.join(data_root, 'audios', 'content', content)
        print(f"\n🔍 Content: {content_path}")

        content_audio_path = os.path.join(content_path, sorted([f for f in os.listdir(content_path) if f.endswith('.wav')])[0])
        content_embed = compute_clap_audio_embedding_from_mel(content_audio_path, processor, clap_model, device)

        for style1, style2 in tqdm(style_pairs, desc=f"{content} Style Pairs", leave=False):
            output_dir = os.path.join(output_root, content, style1 + "_" + style2, 'audios')
            if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
                continue
            interpolated_audio_path = os.path.join(output_dir, sorted([f for f in os.listdir(output_dir) if f.endswith('.wav')])[0])
            interpolated_embed = compute_clap_audio_embedding_from_mel(interpolated_audio_path, processor, clap_model, device)

            style_1_audio_path = os.path.join(style_root, style1, sorted([f for f in os.listdir(os.path.join(style_root, style1)) if f.endswith('.wav')])[0])
            style_2_audio_path = os.path.join(style_root, style2, sorted([f for f in os.listdir(os.path.join(style_root, style2)) if f.endswith('.wav')])[0])
            style_1_embed = compute_clap_audio_embedding_from_mel(style_1_audio_path, processor, clap_model, device)
            style_2_embed = compute_clap_audio_embedding_from_mel(style_2_audio_path, processor, clap_model, device)

            style_1_text_embed = compute_clap_text_embedding(f"A music piece in the style of {style1}", processor, clap_model, device)
            style_2_text_embed = compute_clap_text_embedding(f"A music piece in the style of {style2}", processor, clap_model, device)

            content_preservation = cosine_similarity(content_embed, interpolated_embed)
            style_1_fit = cosine_similarity(style_1_text_embed, interpolated_embed)
            style_2_fit = cosine_similarity(style_2_text_embed, interpolated_embed)

            style_1_ref_fit = cosine_similarity(style_1_text_embed, style_1_embed)
            style_2_ref_fit = cosine_similarity(style_2_text_embed, style_2_embed)

            style_fit_singles[style_1_ref_fit] = style_1_ref_fit
            style_fit_singles[style_2_ref_fit] = style_2_ref_fit
            content_preservations.append(content_preservation)
            style_fits.append((style_1_fit + style_2_fit) / 2)
            single_style_fits.append(max(style_1_fit, style_2_fit))

        for style in tqdm(style_dirs, desc=f"{content} Single Styles", leave=False):
            output_dir = os.path.join(output_root, content, style, 'audios')
            if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
                continue
            interpolated_audio_path = os.path.join(output_dir, sorted([f for f in os.listdir(output_dir) if f.endswith('.wav')])[0])
            interpolated_embed = compute_clap_audio_embedding_from_mel(interpolated_audio_path, processor, clap_model, device)

            style_audio_path = os.path.join(style_root, style, sorted([f for f in os.listdir(os.path.join(style_root, style)) if f.endswith('.wav')])[0])
            style_embed = compute_clap_audio_embedding_from_mel(style_audio_path, processor, clap_model, device)

            style_text_embed = compute_clap_text_embedding(f"A music piece in the style of {style}", processor, clap_model, device)

            content_preservation = cosine_similarity(content_embed, interpolated_embed)
            style_fit = cosine_similarity(style_text_embed, interpolated_embed)
            style_ref_fit = cosine_similarity(style_text_embed, style_embed)

            style_fit_singles[style_ref_fit] = style_ref_fit
            content_preservations_singles.append(content_preservation)
            style_fits_singles.append(style_fit)

    print("\n📊 === Aggregated Metrics ===")
    print(f"Content Preservation (mean): {np.mean(content_preservations):.4f}")
    print(f"Style Fit (mean):             {np.mean(style_fits):.4f}")
    print(f"Single Style Fit (mean):      {np.mean(single_style_fits):.4f}")
    print(f"Content Preservation Singles (mean): {np.mean(content_preservations_singles):.4f}")
    print(f"Style Fit Singles (mean): {np.mean(style_fits_singles):.4f}")

if __name__ == "__main__":
    main()
