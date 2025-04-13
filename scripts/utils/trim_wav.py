from pydub import AudioSegment
import os

def split_wav(input_path: str, output_dir: str, segment_duration: int = 5):
    """
    Splits a WAV file into smaller segments.

    Args:
        input_path (str): Path to the input WAV file.
        output_dir (str): Directory to save the output segments.
        segment_duration (int): Duration of each segment in seconds (default: 5).
    """
    os.makedirs(output_dir, exist_ok=True)

    audio = AudioSegment.from_wav(input_path)
    duration_ms = len(audio)
    segment_ms = segment_duration * 1000

    for i in range(0, duration_ms, segment_ms):
        segment = audio[i:i + segment_ms]
        segment.export(os.path.join(output_dir, f"segment_{i // segment_ms}.wav"), format="wav")

    print(f"Split complete. {duration_ms // segment_ms + 1} segments saved to {output_dir}.")

# Example usage
if __name__ == "__main__":
    input_wav = "/home/dcor/shaiyehezkel/MusicTI_AAAI2024/images/style/girl/girl_la_la.wav"
    output_folder = "/home/dcor/shaiyehezkel/MusicTI_AAAI2024/images/style/girl_split"
    split_wav(input_wav, output_folder, segment_duration=5)
