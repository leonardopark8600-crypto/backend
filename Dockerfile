FROM python:3.11-slim

RUN apt-get update && apt-get install -y     build-essential     libxml2     libxslt1.1     libgl1     libglu1     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
