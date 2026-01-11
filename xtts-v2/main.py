import torch
from TTS.tts.configs.xtts_config import XttsConfig
torch.serialization.add_safe_globals([XttsConfig])
from TTS.api import TTS


with open("../target_texts.txt", "r") as f:
    target_texts = f.readlines()
target_texts = [text.strip() for text in target_texts]

with open("../reference_text.txt", "r") as f:
    reference_text = f.read().strip()

model_names = ["xtts_v2"]

for model_name in model_names:
    tts = TTS(f"tts_models/multilingual/multi-dataset/{model_name}", gpu=True)
    for idx, target_text in enumerate(target_texts):
        tts.tts_to_file(
            text=target_text,
            file_path=f"../synthetic-audios/{model_name}-{idx:03d}.wav",
            speaker_wav="../reference_audio.wav",
            language="en"
        )