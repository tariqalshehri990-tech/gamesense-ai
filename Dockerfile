FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY nafas ./nafas
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn nafas.server:app --host 0.0.0.0 --port ${PORT}"]
