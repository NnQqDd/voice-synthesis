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
        "python3", "inference.py",
        "--source", source_audio,
        "--target", target_audio,
        "--output", "../voice-conversion-audios",
        "--diffusion-steps", "25",
        "--length-adjust", "1.0",
        "--inference-cfg-rate", "0.7",
        "--f0-condition", "False",
        "--auto-f0-adjust", "False",
        "--semi-tone-shift", "0",
        "--fp16", "True",
        "--index", str(idx),
    ]
    subprocess.run(cmd, check=True)


for idx, source_audio in enumerate(source_audios):    
    cmd = [
        "python3", "inference_v2.py",
        "--source", source_audio,
        "--target", target_audio,
        "--output", "../voice-conversion-audios",
        "--diffusion-steps", "25",
        "--length-adjust", "1.0",
        "--intelligibility-cfg-rate", "0.7",
        "--similarity-cfg-rate", "0.7",
        "--convert-style", "True",
        "--anonymization-only", "False",
        "--top-p", "0.9",
        "--temperature", "1.0",
        "--repetition-penalty", "1.0",
        "--index", str(idx),
    ]
    subprocess.run(cmd, check=True)