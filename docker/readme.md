# Doker — Build & run 
Minimal steps to build the image and run it with secrets.

Prerequisites
- Docker installed
- A Dockerfile in the current directory
- A secret file (e.g. `.env` or `secret.txt`)

Build
```sh
docker build -t agentcore:latest -f docker/dockerfile .
```

Run using an environment file
1. Create `.env` (e.g. `SECRET_KEY=...`)
2. Run:
```sh
docker run -p 8000:8000 --env-file .env agentcore
```

That's all.