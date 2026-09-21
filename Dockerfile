FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1


WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements.txt


COPY . .

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
    "--server.adress=0.0.0.0", "--server.port=8501"]