FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ models/ /app/
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
