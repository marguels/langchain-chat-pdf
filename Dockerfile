FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

# Copying requirements first will help with cache
COPY ./requirements.txt /app

RUN pip install --upgrade -r requirements.txt
# RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./main.py /app
COPY ./.env /app

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]