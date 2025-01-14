ARG RESOURCE_PATH=../resources

# Building stage
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Finalizing stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy source code
COPY . .

# Ensure paths are correct for copying resources and scripts

# COPY scripts /app/scripts
# COPY ${RESOURCE_PATH}/data/store_processed.csv /app/resources/store.csv
# COPY ${RESOURCE_PATH}/scalers/scaler.pkl /app/resources/scaler.pkl
# COPY ${RESOURCE_PATH}/encoders/onehot_encoder.pkl /app/resources/encoder.pkl
# COPY ${RESOURCE_PATH}/models/lgb_model.pkl /app/resources/sales_model.pkl
# COPY ${RESOURCE_PATH}/models/ltsm_sales_model.h5 /app/resources/sales_model.h5

# EXPOSE 7777
# ENV NAME env
# CMD ["python", "main.py"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7777"]
# CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "7777"]

# docker build -t sales-api .
# docker run -p 7777:7777 sales-api
# docker build --build-arg RESOURCE_PATH=../resources SCRIPTS_PATH=../scripts -t sales-api .