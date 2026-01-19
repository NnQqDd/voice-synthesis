import os
from tqdm import tqdm
from TTS.api import TTS

# Load the model (GPU recommended for speed)
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False, gpu=True)  # gpu=False if no CUDA

source_path = "/home/duyn/ActableDuy/datasets/VCTK/VCTK-Corpus/VCTK-Corpus/wav48/p225"
source_audios = [os.path.join(source_path, filename) for filename in os.listdir(source_path)]
source_audios = source_audios[:200]
target_audio = "../reference_audio.wav"
# Perform voice conversion

for idx, source_audio in tqdm(enumerate(source_audios)):
    tts.voice_conversion_to_file(
        source_wav=source_audio,  # Audio with the content/prosody to keep
        target_wav=target_audio,  # Reference audio for the desired voice
        file_path=f"../voice-conversion-audios/freevc-{idx:03d}.wav"
    )