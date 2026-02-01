import subprocess
import os


source_path = "/home/duyn/ActableDuy/datasets/VCTK/VCTK-Corpus/VCTK-Corpus/wav48/p225"
source_audios = [os.path.join(source_path, filename) for filename in os.listdir(source_path)]
source_audios.sort()
source_audios = source_audios[:200]
print(source_audios)
target_audio = "../reference_audio.wav"

for idx, source_audio in enumerate(source_audios):    
    cmd = [
        "python", "convert.py",
        source_audio,   # source utterance
        target_audio,
        "-w", "wav2vec_small.pt",
        "-v", "vocoder.pt",
        "-c", "fragmentvc.pt",
        "-o", "../voice-conversion-audios/fragment_vc-{idx:03d}.wav".format(idx=idx)
    ]
    subprocess.run(cmd, check=True)
