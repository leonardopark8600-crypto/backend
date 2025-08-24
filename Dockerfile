# Usa Python slim para reducir tamaño de imagen
FROM python:3.11-slim

# Evita prompts interactivos en instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependencias del sistema necesarias para tellurium, libsbml, scipy, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2 \
    libxml2-dev \
    libxslt1-dev \
    libsbml5-dev \
    && rm -rf /var/lib/apt/lists/*

# Carpeta de trabajo
WORKDIR /app

# Copia requirements primero (para aprovechar cache)
COPY requirements.txt /app/

# Instala dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia TODO el código del proyecto, incl. opt_addon/
COPY . /app

# Exponer puerto
EXPOSE 8000

# Comando por defecto: correr la app con uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
