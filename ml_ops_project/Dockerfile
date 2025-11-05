# ==============================
# 🌟 Stage 1 — Base Environment
# ==============================
FROM python:3.11-slim

# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies (for numpy, pandas, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ==============================
# 🌟 Stage 2 — Run Application
# ==============================

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
