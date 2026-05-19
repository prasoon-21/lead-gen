FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api/ ./api/
COPY core/ ./core/
COPY config/ ./config/
COPY workflows/ ./workflows/

# Create logs directory
RUN mkdir -p logs

# Railway assigns PORT dynamically, default to 8000 for local
ENV PORT=8000
EXPOSE $PORT

# Run the API server using shell form to allow variable expansion
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
