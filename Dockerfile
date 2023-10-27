FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY . /app

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]