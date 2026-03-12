import runpod
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to(device)

def handler(job):

    text = job["input"]["text"]

    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        speech = model.generate(**inputs)

    return {
        "audio": speech.cpu().numpy().tolist()
    }

runpod.serverless.start({"handler": handler})
