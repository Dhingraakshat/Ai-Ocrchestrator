FROM python:3.11-slim

# System deps for plyer (Linux desktop notifications) and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnotify-bin \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories that agents write to at runtime
RUN mkdir -p memory_store

EXPOSE 8000

CMD ["python", "main.py"]
