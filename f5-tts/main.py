from f5_tts.api import F5TTS

tts_model = F5TTS()
with open("../target_texts.txt", "r") as f:
    target_texts = f.readlines()
target_texts = [text.strip() for text in target_texts]

with open("../reference_text.txt", "r") as f:
    reference_text = f.read().strip()

model_names = ['E2TTS_Base', 'F5TTS_v1_Base']

for model_name in model_names:
    tts_model = F5TTS(model=model_name)
    for nfe_step in [16, 32, 64]:
        for idx, target_text in enumerate(target_texts):
            wav, sr, spec = tts_model.infer(
                ref_file="../reference_audio.wav",  # Path to reference WAV file (voice to clone)
                ref_text=reference_text,  # Or "" for auto-transcription
                gen_text=target_text,  # Text to synthesize
                file_wave=f"../voice-clone-audios/{model_name.lower()}_{nfe_step}-{idx:03d}.wav",  # Path to save generated WAV
                file_spec=None,  # Optional: Path to save spectrogram image
                nfe_step=nfe_step,
                seed=42,  # Optional: For reproducibility
            ) # nfe_step=32,