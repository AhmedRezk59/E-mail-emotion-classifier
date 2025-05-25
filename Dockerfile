# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 7860

# Set Streamlit config (to avoid prompt for sharing telemetry)
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]
