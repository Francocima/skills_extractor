FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model
RUN python -m spacy download en_core_web_sm


COPY . .

# Expose port for the API
EXPOSE 8080

# Command to run the API
CMD ["uvicorn", "skills_extractor:app", "--host", "0.0.0.0", "--port", "8080"]
