FROM python:3.10-slim-bookworm

WORKDIR /PentionSystem

# COPY PentionSystem/requirements.txt .
# Installare dipendenze di sistema minime
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
COPY PentionSystem/pyproject.toml PentionSystem/poetry.lock* ./

# Installa dipendenze senza creare virtualenv
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main --no-root


#RUN pip install --no-cache-dir -r requirements.txt --verbose

COPY PentionSystem/ .
COPY gaussianPuff/ ./gaussianPuff/


EXPOSE 8004

CMD ["python", "-m", "streamlit", "run", "application.py", "--server.port=8004", "--server.address=0.0.0.0"]
