FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISTILBERT_PATH=jinha1491/poker-distilbert

# CPU-only torch. The default Linux wheel ships with CUDA libraries (several GB)
# that a CPU container never uses.
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# pull the fine-tuned model at build time so the container starts fast
RUN python -c "from transformers import DistilBertTokenizer, DistilBertModel; \
DistilBertTokenizer.from_pretrained('jinha1491/poker-distilbert'); \
DistilBertModel.from_pretrained('jinha1491/poker-distilbert')"

EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
