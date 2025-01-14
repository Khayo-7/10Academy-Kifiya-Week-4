# Building stage
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Finalizing stage
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY . .

# Copy specific scripts from the scripts folder
COPY ../scripts /app/scripts

# Copy and rename resource files
COPY ../resources/data/store_processed.csv /app/resources/data/store.csv
COPY ../resources/encoders/onehot_encoder.pkl /app/resources/encoders/encoder.pkl
COPY ../resources/scalers/scaler.pkl /app/resources/scalers/scaler.pkl

# Copy the model files
COPY  ../resources/models/ltsm_sales_model.pkl /app/models/sales_model.pkl
COPY  ../resources/models/ltsm_sales_model.hs /app/models/sales_model.hs

# EXPOSE 7777
# ENV NAME env
# CMD ["python", "main.py"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7777"]
# CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "7777"]

# docker build -t sales-api .
# docker run -p 7777:7777 sales-api
