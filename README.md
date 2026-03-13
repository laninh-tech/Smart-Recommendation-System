# SmartRec Intelligence Platform

Production-minded recommendation system with an MLOps workflow, model serving, and analytics dashboard.

## Project Objective
Build an end-to-end recommendation pipeline that can be trained, evaluated, and served in a reproducible way for e-commerce-style ranking use cases.

## Tech Stack
- Python (training and inference)
- TypeScript + React (dashboard)
- Docker and Docker Compose (local orchestration)
- Data tooling for ETL and feature preparation

## Architecture Overview
- Data preparation and feature engineering module
- Model training module (candidate ranking models)
- Inference/API layer for online recommendations
- Frontend dashboard for monitoring and analysis

## Installation
```bash
git clone https://github.com/laninh-tech/smartrec-dashboard.git
cd smartrec-dashboard
npm install
pip install -r backend/requirements.txt
```

## Run
```bash
# Option 1: full stack with containers
docker compose up --build

# Option 2: manual startup
npm run dev
```

## Current Results
- Built a complete train-to-inference workflow in one repository.
- Added dashboard visibility for recommendation behavior analysis.
- Established a baseline structure for CI/CD and model versioning.

## Roadmap
- Add offline metrics dashboard (Recall@K, NDCG@K).
- Introduce experiment tracking and model registry.
- Add online A/B evaluation hooks.

## Author
La Quang Ninh