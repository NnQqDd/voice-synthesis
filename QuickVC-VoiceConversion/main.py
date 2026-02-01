import subprocess
import os

# Write audios to txt
source_path = "/home/duyn/ActableDuy/datasets/VCTK/VCTK-Corpus/VCTK-Corpus/wav48/p225"
source_audios = [os.path.join(source_path, filename) for filename in os.listdir(source_path)]
source_audios.sort()
source_audios = source_audios[:200]
print(source_audios)
target_audio = "../reference_audio.wav"

with open("convert.txt", "w") as f:
    for idx, source_audio in enumerate(source_audios):
        f.write(f"QuickVC-{idx:03d}|{source_audio}|{target_audio}\n")

cmd = [
    "python3", "convert.py",
    "--hpfile", "logs/quickvc/config.json",
    "--ptfile", "logs/quickvc/G_1200000.pth",
    "--txtpath", "convert.txt",
    "--outdir", "../voice-conversion-audios",
]

subprocess.run(cmd, check=True)