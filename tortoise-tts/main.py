import numpy as np
import soundfile as sf
from tortoise import api
from tortoise.utils.audio import load_audio

with open("../target_texts.txt", "r") as f:
    target_texts = f.readlines()
target_texts = [text.strip() for text in target_texts]

paths = ["../reference_audio.wav"]
clips = [load_audio(p, 22050) for p in paths]

tts = api.TextToSpeech()

for preset in ["ultra_fast", "fast", "standard"]:
    for idx, target_text in enumerate(target_texts):
        pcm_audio = tts.tts_with_preset(target_text, voice_samples=clips, preset=preset)
        sf.write(f"../voice-clone-audios/tortoise_tts_{preset}-{idx:03d}.wav", np.squeeze(pcm_audio), 22050)
