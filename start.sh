#!/bin/bash

langgraph dev --host 0.0.0.0 --port 8123 &

uvicorn app:app --host 0.0.0.0 --port 8000
