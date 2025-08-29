FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y git ffmpeg build-essential zlib1g && \
    pip install --upgrade pip

RUN pip install git+https://github.com/openai/whisper.git && \
    pip install faster-whisper==1.* ctranslate2==4.* && \
    pip install pyannote.audio ffmpeg-python

WORKDIR /app

COPY run.py .
COPY audio ./audio

CMD ["python", "run.py"]
