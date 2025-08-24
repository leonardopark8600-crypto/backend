FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Instala librerías necesarias para tellurium/libsbml/antimony
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2 \
    libxml2-dev \
    libxslt1-dev \
    libsbml5-dev \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia requirements primero para aprovechar cache
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código, incluyendo opt_addon/
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
