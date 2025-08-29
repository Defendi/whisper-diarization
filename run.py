import torch
import os
from pyannote.audio import Pipeline

AUDIO_FILE = os.environ.get("AUDIO_FILE")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
FASTER_WHISPER = os.environ.get("FASTER_WHISPER", False) == "true"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", False) or "medium"
FASTER_WHISPER_COMPUTE = os.getenv("FASTER_WHISPER_COMPUTE", "")
FASTER_WHISPER_BEAM = int(os.getenv("FASTER_WHISPER_BEAM", "5"))
PYANNOTE_MIN_SPK = os.getenv("PYANNOTE_MIN_SPK", False)
PYANNOTE_MAX_SPK = os.getenv("PYANNOTE_MAX_SPK", False)
RUN_WITH_GPU = os.getenv("RUN_WITH_GPU", "true") == "true"

print(f"🎧 Arquivo de áudio: {AUDIO_FILE}")
print(f"💾 Arquivo de saída: {OUTPUT_FILE}")
print(f"🔑 Token do Hugging Face: {HUGGINGFACE_TOKEN}")
print(f"🤖 Modelo Whisper: {WHISPER_MODEL}")
print(f"⚡ Usar FASTER-WHISPER: {FASTER_WHISPER}")
if FASTER_WHISPER:
    print(f"   - compute_type: {FASTER_WHISPER_COMPUTE or 'padrão'}")
    print(f"   - beam_size: {FASTER_WHISPER_BEAM}")


if not HUGGINGFACE_TOKEN:
    raise ValueError("❌ Token do Hugging Face não encontrado. Defina HUGGINGFACE_TOKEN.")

# Vrificação de GPU
if RUN_WITH_GPU:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Dispositivo: {device}")
else:
    device = "cpu"
    print("🖥️ Forçando uso de CPU")

if device == "cpu":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
segments_out = []

if FASTER_WHISPER:
    print("🚀 Usando FASTER-WHISPER")
    from faster_whisper import WhisperModel

    if not FASTER_WHISPER_COMPUTE:
        FASTER_WHISPER_COMPUTE = "float16" if device == "cuda" else "int8"

    print(f"[fw] compute_type={FASTER_WHISPER_COMPUTE} beam={FASTER_WHISPER_BEAM}")
    fw_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=FASTER_WHISPER_COMPUTE)
    fw_segments, info = fw_model.transcribe(
        AUDIO_FILE,
        language="pt",
        vad_filter=True,
        beam_size=FASTER_WHISPER_BEAM,
    )
    for s in fw_segments:
        segments_out.append({"start": s.start, "end": s.end, "text": s.text})

else:
    print("🚀 Usando OPENAI WHISPER")
    import whisper

    # Transcrição com Whisper
    print(f"🔤 Transcrevendo com Whisper no {device}...")
    model = whisper.load_model(WHISPER_MODEL, device=device)
    result = model.transcribe(AUDIO_FILE,
                            task="transcribe",
                            fp16=(device=="cuda"),
                            language="pt",
                            temperature=0.0)
    segments_out = result.get("segments", [])

# Diarização com PyAnnote
print(f"🔊 Realizando diarização com PyAnnote... ({HUGGINGFACE_TOKEN})")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGINGFACE_TOKEN)

# Força uso de GPU
if device == "cuda":
    print("🚀 Usando GPU para diarização")
    pipeline.to(torch.device("cuda"))

kwargs = {}
if PYANNOTE_MIN_SPK:
    kwargs["min_speakers"] = int(PYANNOTE_MIN_SPK)
if PYANNOTE_MAX_SPK:
    kwargs["max_speakers"] = int(PYANNOTE_MAX_SPK)

# Realizando diarização
print("🎤 Processando áudio para diarização...")
diarization = pipeline(AUDIO_FILE, **kwargs)

# Combinação dos dados
print("🧠 Combinando falas com falantes...")
# ================= Merge speaker + text =================
def pick_speaker(t, ann):
    for (segment, _, label) in ann.itertracks(yield_label=True):
        if segment.start <= t <= segment.end:
            return label
    return "speaker_?"

print("[merge] combinando…")
final_lines = []
for seg in segments_out:
    text = (seg.get("text") or "").strip()
    if not text:
        continue
    spk = pick_speaker(seg["start"], diarization)
    final_lines.append(f"[{spk}] {text}")

# ================= Save =================
os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(final_lines))

print(f"✅ Transcrição com falantes salva em: {OUTPUT_FILE}")
