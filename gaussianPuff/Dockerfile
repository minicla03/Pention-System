FROM python:3.10-slim-bookworm

WORKDIR /gaussianPuff

#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt --verbose

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Aggiorna pip
RUN pip install --no-cache-dir --upgrade pip

# Installa Poetry usando lo script ufficiale
RUN curl -sSL https://install.python-poetry.org | python3 -

# Aggiungi Poetry al PATH
ENV PATH="/root/.local/bin:$PATH"

# Copia pyproject.toml e poetry.lock
COPY pyproject.toml poetry.lock* ./


# Installa dipendenze senza creare virtualenv
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main --no-root

COPY . .

EXPOSE 8002

CMD ["uvicorn", "api_gaussian:app", "--host", "0.0.0.0", "--port", "8002"]
