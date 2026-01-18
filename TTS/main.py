import os
import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("../target_texts.txt", "r") as f:
    target_texts = f.readlines()
target_texts = [text.strip() for text in target_texts]

MODEL_NAMES = ["tts_models/multilingual/multi-dataset/xtts_v2", "tts_models/multilingual/multi-dataset/your_tts"]

for model_name in MODEL_NAMES:
    tts = TTS(model_name).to(device)
    for idx, target_text in enumerate(target_texts):    
        tts.tts_to_file(
            text=target_text,
            speaker_wav="../reference_audio.wav",
            language="en",
            file_path=f"../voice-clone-audios/{os.path.basename(model_name)}-{idx:03d}.wav"
        )
