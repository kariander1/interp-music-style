import os
from pathlib import Path
from tqdm import tqdm
from scripts.utils.cli import image_to_audio

def convert_all_images_to_audio(root_dir):
    root_dir = Path(root_dir)
    png_files = list(root_dir.rglob("*.png"))  # Recursively find all PNG files

    print(f"🔍 Found {len(png_files)} PNG files in {root_dir}")

    for img_path in tqdm(png_files, desc="Converting images to audio"):
        # Determine output path: go two levels up, then into "audios/"
        relative_path = img_path.relative_to(root_dir)
        output_audio_path = root_dir / relative_path.parent.parent / "audios" / relative_path.with_suffix(".wav").name

        # Skip if audio already exists
        if output_audio_path.exists():
            continue

        # Ensure output directory exists
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert image to audio
        image_to_audio(image=str(img_path), audio=str(output_audio_path))

    print("✅ All conversions done.")

# Example usage
if __name__ == "__main__":
    convert_all_images_to_audio("outputs")
