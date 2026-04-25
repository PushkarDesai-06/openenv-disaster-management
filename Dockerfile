FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY disaster_sim /app/disaster_sim

EXPOSE 7860

CMD ["uvicorn", "disaster_sim.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
