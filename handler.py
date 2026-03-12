import runpod
import torch
from transformers import AutoProcessor, AutoModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to(device)

def handler(job):
    text = job["input"]["text"]

    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        audio = model.generate(**inputs)

    audio = audio.cpu().numpy().tolist()

    return {
        "audio": audio
    }

runpod.serverless.start({"handler": handler})
