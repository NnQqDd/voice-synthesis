import torch
import soundfile as sf
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel


spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")

with torch.no_grad():
    parsed = spec_generator.parse("You can type your sentence here to get nemo to produce speech.")
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    audio = model.convert_spectrogram_to_audio(spec=spectrogram)

sf.write("speech.wav", audio.squeeze().cpu().float().numpy(), 22050)