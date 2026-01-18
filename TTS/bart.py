import torch
from TTS.api import TTS

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)

# Save to file
tts.tts_to_file(
    text="Hello World, I am not a cafe!",
    speaker_wav="../reference_audio.wav",
    language="en",
    file_path="output.wav"
)