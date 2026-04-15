FROM python:3.12-slim

WORKDIR /app

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY docker/flow_config.json .

WORKDIR /app/src

CMD ["python", "run_flow.py"]
