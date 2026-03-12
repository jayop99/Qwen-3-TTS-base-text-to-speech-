import runpod
from transformers import pipeline

# Load Qwen TTS model
tts = pipeline(
    "text-to-speech",
    model="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)

def handler(job):
    text = job["input"]["text"]

    result = tts(text)

    return result

runpod.serverless.start({"handler": handler})
