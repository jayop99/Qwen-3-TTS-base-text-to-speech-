import runpod
import torch
from transformers import AutoProcessor, AutoModel

model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def handler(job):
    text = job["input"]["text"]

    inputs = processor(text=text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        audio = model.generate(**inputs)

    audio = audio.cpu().numpy().tolist()

    return {"audio": audio}

runpod.serverless.start({"handler": handler})
