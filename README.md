# 🧪 Lab: Chat with your PDFs
This application is an ongoing experimentat to test and try out new LLM technologies. Its purpose it's purely educational.

Currently, it's used to upload PDF documents and create the embeddings which will be used to offer a _context window_ for the chosen LLM text-generator.

The generated embeddings are stored in a Qdrant Vector database.

## How to run
1. Build the docker images
```sh
docker-compose build
```

2. Run
```sh
docker-compose up
```