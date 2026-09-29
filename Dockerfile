FROM python:3.12-slim

WORKDIR /code
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# CPU-only PyTorch (much smaller than the default CUDA build)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY model ./model

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]