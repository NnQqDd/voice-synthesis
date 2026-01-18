from TTS.api import TTS

# Load the model (GPU recommended for speed)
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False, gpu=True)  # gpu=False if no CUDA

# Perform voice conversion
tts.voice_conversion_to_file(
    source_wav="../reference_audio.wav",  # Audio with the content/prosody to keep
    target_wav="path/to/target_voice.wav",  # Reference audio for the desired voice
    file_path="output_converted.wav"       # Output file
)