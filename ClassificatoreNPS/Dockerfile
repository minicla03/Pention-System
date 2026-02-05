FROM python:3.10-slim-bookworm

WORKDIR /ClassificatoreNPS

#COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Aggiorna pip
RUN pip install --no-cache-dir --upgrade pip

# Installa Poetry usando lo script ufficiale
RUN pip install --no-cache-dir poetry


# Aggiungi Poetry al PATH
ENV PATH="/root/.local/bin:$PATH"

# Copia pyproject.toml e poetry.lock
COPY pyproject.toml poetry.lock* ./


# Installa dipendenze senza creare virtualenv
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main --no-root


#RUN pip install --no-cache-dir -r requirements.txt --verbose

COPY . .
COPY model ./model

EXPOSE 8000

CMD ["uvicorn", "api_clf_nps:app", "--host", "0.0.0.0", "--port", "8000"]
