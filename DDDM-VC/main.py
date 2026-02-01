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
        "--src_path", source_audio,
        "--trg_path", target_audio,
        "--ckpt_model", "./ckpt/model_base.pth",
        "--ckpt_voc", "./vocoder/voc_ckpt.pth",
        "--ckpt_f0_vqvae", "./f0_vqvae/f0_vqvae.pth",
        "--output_dir", "../voice-conversion-audios",
        "--index", str(idx),
        "-t", "6",
    ]

    subprocess.run(cmd, check=True)