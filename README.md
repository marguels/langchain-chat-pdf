# How to run
## With Docker
1. Build the docker image
```sh
docker build -t streamlit_app .
```

2. Run specifying the `.env` file
```sh
docker run -p 8080:8080 --env-file .env streamlit_app
```

## Locally

1. Create virtual environment using `venv` module:
```sh
python3 -m venv .venv
```
2. Activate virtual environment:
```sh
source .venv/bin/activate
```
Make sure you have selected the correct interpreter in VSCode.

3. Install all requirements:
```sh
pip install -r requirements.txt
```

4. Create an `.env` file at the root folder and add your `HUGGINGFACEHUB_API_TOKEN`

5. Run the app
```sh
streamlit run main.py
```