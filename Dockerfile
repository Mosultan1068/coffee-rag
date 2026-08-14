# Small, official Python base image
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first, so Docker can cache the pip install step
# separately from code changes (same trick as the fault-finding project)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Gradio's default port, so it's reachable from outside the container
EXPOSE 7860

# Run the Gradio app. Binding to 0.0.0.0 (not the default 127.0.0.1)
# is required inside a container, so it accepts connections from
# outside the container, not just from within it.
CMD ["python", "app.py"]
