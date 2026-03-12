import runpod
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def handler(job):
    text = job["input"]["text"]

    inputs = processor(text=text, return_tensors="pt").to(model.device)

    speech = model.generate(**inputs)

    return {"audio": speech.tolist()}

runpod.serverless.start({"handler": handler})
