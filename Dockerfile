FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y git ffmpeg build-essential && \
    pip install --upgrade pip

RUN pip install git+https://github.com/openai/whisper.git && \
    pip install faster-whisper && \
    pip install pyannote.audio ffmpeg-python

WORKDIR /app

COPY run.py .
COPY .env .
COPY audio ./audio

CMD ["python", "run.py"]
