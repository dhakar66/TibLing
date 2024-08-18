

from transformers import pipeline
import os

# Ensure ffmpeg is installed and available in PATH
# Check if ffmpeg is properly installed
# os.system("ffmpeg -version")

# Load your model
pipe = pipeline("automatic-speech-recognition", model="openpecha/wav2vec2_run9")

# List of audio files
audio_files = [
    "Cat.m4a",
    "Dragon.m4a",
    "Human.m4a",
    "School.m4a",
    "Two.m4a",
    "Bird.wav"
]

# Process each audio file
for audio_file in audio_files:
    if os.path.isfile(audio_file):
        try:
            res = pipe(audio_file)
            print(f"Result for {audio_file}: {res}")
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    else:
        print(f"File not found: {audio_file}")
