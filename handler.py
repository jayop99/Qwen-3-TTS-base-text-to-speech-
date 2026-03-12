import runpod
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen3TTSModel.from_pretrained(
    MODEL_ID,
    device_map=device,
    dtype=torch.bfloat16
)

def handler(job):

    text = job["input"]["text"]

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English"
    )

    return {
        "audio": wavs[0].tolist(),
        "sample_rate": sr
    }

runpod.serverless.start({"handler": handler})
