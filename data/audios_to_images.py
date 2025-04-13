from pathlib import Path
from scripts.utils.cli import audio_to_image
from tqdm import tqdm
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
    Recursively converts all .wav files in a directory to images using audio_to_image,
    preserving the directory structure in the output directory.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)

    for audio_file in tqdm(audio_dir.rglob("*.wav")):
        # Compute the relative path of the audio file with respect to the input dir
        relative_path = audio_file.relative_to(audio_dir)
        # Change the suffix to .png and use it as the output file path
        image_file = output_dir / relative_path.with_suffix(".png")
        # Make sure the parent directory exists
        image_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing {audio_file} → {image_file}")
        audio_to_image(audio=str(audio_file), image=str(image_file), **params)


if __name__ == "__main__":
    input_audio_dir = "data/musicTI_dataset/audios"
    output_image_dir = "data/musicTI_dataset/images"
    process_wav_directory(input_audio_dir, output_image_dir)