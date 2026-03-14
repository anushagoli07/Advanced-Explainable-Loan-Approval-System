# Base Image
FROM python:3.10-slim

# Working Directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project Files
COPY . .

# Expose ports: 8000 (FastAPI), 8501 (User UI), 8502 (Admin UI)
EXPOSE 8000 8501 8502

# Startup Script (Simplified for Docker)
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.1 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 & streamlit run frontend/admin_dashboard.py --server.port 8502 --server.address 0.0.0.0"]
