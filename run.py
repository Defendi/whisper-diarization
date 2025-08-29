import whisper
import torch
import os
from pyannote.audio import Pipeline

AUDIO_FILE = "audio/Primeira_reunião_27-08-2025 18.48.mp3"
OUTPUT_FILE = "output/Primeira_reunião_com_falantes.txt"

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("❌ Token do Hugging Face não encontrado. Defina HUGGINGFACE_TOKEN.")

# Vrificação de GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transcrição com Whisper
print(f"🔤 Transcrevendo com Whisper no {device}...")
model = whisper.load_model("medium", device=device)
result = model.transcribe(AUDIO_FILE,
                          task="transcribe",
                          fp16=(device=="cuda"),
                          language="pt",
                          temperature=0.0)

# Diarização com PyAnnote
print(f"🔊 Realizando diarização com PyAnnote... ({HUGGINGFACE_TOKEN})")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGINGFACE_TOKEN)

# Força uso de GPU
if device == "cuda":
    print("🚀 Usando GPU para diarização")
    pipeline.to(torch.device("cuda"))

# Realizando diarização
print("🎤 Processando áudio para diarização...")
diarization = pipeline(AUDIO_FILE)

# Combinação dos dados
print("🧠 Combinando falas com falantes...")
def get_speaker(time, diarization):
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if segment.start <= time <= segment.end:
            return speaker
    return "Desconhecido"

final_text = ""
for segment in result["segments"]:
    speaker = get_speaker(segment["start"], diarization)
    text = segment["text"].strip()
    final_text += f"[{speaker}] {text}\n"

# Salvando saída
os.makedirs("output", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(final_text)

print(f"✅ Transcrição com falantes salva em: {OUTPUT_FILE}")
