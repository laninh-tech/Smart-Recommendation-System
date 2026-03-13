# SmartRec Dashboard

MLOps-oriented recommendation system project with deep learning models and an analytics dashboard for experimentation and serving.

## Overview
SmartRec focuses on building a full recommendation workflow: data preparation, model training, inference service, and visualization layer for monitoring recommendation quality.

## Core modules
- Data pipeline for preparing recommendation datasets.
- Deep learning recommendation models for ranking/prediction.
- Backend service for inference and API integration.
- Frontend dashboard for result inspection and analysis.
- Docker-based local orchestration.

## Tech stack
- Python (modeling, training, serving)
- TypeScript + React (dashboard)
- Docker / Docker Compose (environment orchestration)
- SQL/Data tooling for data processing

## Project structure
```text
smartrec-dashboard/
|-- backend/
|-- src/
|-- docs/
|-- Dockerfile
|-- docker-compose.yml
|-- server.ts
|-- package.json
```

## Quick start
```bash
git clone https://github.com/laninh-tech/smartrec-dashboard.git
cd smartrec-dashboard
npm install
pip install -r backend/requirements.txt
# run app stack (choose one based on your environment)
docker compose up --build
# or start services manually
npm run dev
```

## MLOps direction
- Version datasets and model artifacts.
- Add offline/online evaluation dashboards.
- Automate training and deployment using CI/CD workflows.
- Introduce A/B testing for recommendation strategies.

## Author
La Quang Ninh  
GitHub: https://github.com/laninh-tech