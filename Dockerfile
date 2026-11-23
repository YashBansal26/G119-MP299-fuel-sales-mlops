FROM python:3.12-slim

WORKDIR /service
ENV PYTHONPATH=/service/src \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY src/ src/
COPY models/model.joblib models/model.joblib

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
