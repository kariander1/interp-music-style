import os
from pathlib import Path
from scripts.utils.cli import audio_to_image

# Parameters for audio-to-image conversion
params = dict(
    step_size_ms=10,
    num_frequencies=512,
    min_frequency=0,
    max_frequency=10000,
    stereo=False
)

def process_wav_directory(audio_dir: str, output_dir: str):
    """
    Converts all .wav files in a directory to images using audio_to_image.

    Args:
        audio_dir (str): Path to the directory containing .wav files.
        output_dir (str): Path to the directory to save generated images.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in audio_dir.glob("*.wav"):
        image_file = output_dir / (audio_file.stem + ".png")
        print(f"Processing {audio_file} → {image_file}")
        audio_to_image(audio=str(audio_file), image=str(image_file), **params)

# Example usage
if __name__ == "__main__":
    input_audio_dir = "audios/style/bird"
    output_image_dir = "images/style/bird"
    process_wav_directory(input_audio_dir, output_image_dir)
