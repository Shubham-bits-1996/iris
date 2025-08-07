FROM python:3.9-slim

# 1. Set working dir
WORKDIR /app

# 2. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy code & model
COPY app ./app
COPY model ./model

# 4. Expose port and start app
EXPOSE 80
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]