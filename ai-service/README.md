# AI APK Generator

A Python **3.12** backend service for AI-powered application generation built with **FastAPI**, **Celery**, and asynchronous processing.

The project uses **Poetry** for dependency management and **Docker Compose** for infrastructure services.

---

## 📦 Project Overview

This repository contains two main components:

```
ai-apk-generator/
│
├── infra/        # Infrastructure services (Docker Compose)
│   └── .env
│
├── ai-service/   # Main FastAPI + Celery application
│   └── .env
│
└── README.md
```

### Services Used

* **FastAPI** — API backend
* **Celery** — Background task processing
* **Redis** — Message broker & result backend
* **PostgreSQL** — Database
* **Flower** — Celery monitoring UI
* **Poetry** — Python dependency management
* **Docker Compose** — Infrastructure orchestration

---

## ✅ Prerequisites

Make sure the following are installed and available in your system PATH:

* Python **3.12**
* Docker & Docker Compose
* Poetry

### Install Poetry (if not installed)

```bash
pip install poetry
```

Verify installation:

```bash
poetry --version
```

---

## ⚙️ Environment Configuration

⚠️ The project uses **two separate `.env` files**:

| Location          | Purpose                      |
| ----------------- | ---------------------------- |
| `infra/.env`      | Infrastructure configuration |
| `ai-service/.env` | Application configuration    |

You **must configure both** before running the system.

---

## 🧱 Infrastructure Environment (`infra/.env`)

```env
# --- Database ---
POSTGRES_DB=appdb
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_PORT=5432
POSTGRES_HOST=host.docker.internal

# --- Redis ---
REDIS_PASSWORD=
REDIS_PORT=6379

# --- Application ---
API_PORT=8000
FLOWER_PORT=5555

# --- Celery / Flower ---
CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@redis:6379/0
DATABASE_URL=postgresql+asyncpg://admin:${POSTGRES_PASSWORD}@host.docker.internal:5432/appdb
```

---

## 🤖 AI Service Environment (`ai-service/.env`)

```env
# -------- AI App Builder Service Config --------
APP_NAME=AI App Builder Service
APP_VERSION=0.1.0
APP_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

# App Processing Config
MAX_RETRIES=3
RETRY_DELAY=2

# LLM API Keys
LLAMA3_API_KEY=
LLAMA3_API_BASE=https://fastchat.ideeza.com/v1/chat/completions

FASTCHAT_API_KEY=
```

---

## 🚀 Getting Started

After cloning the repository:

```bash
git clone <repository-url>
cd ai-apk-generator
```

---

## 1️⃣ Start Infrastructure Services

Navigate to the infrastructure directory:

```bash
cd infra
```

Start all required services:

```bash
docker compose up -d
```

This will start:

* PostgreSQL
* Redis
* Flower
* Supporting containers

Wait until all containers are running successfully.

You can verify with:

```bash
docker compose ps
```

---

## 2️⃣ Install Application Dependencies

Open a **new terminal** and navigate to the application service:

```bash
cd ai-service
```

Install Python dependencies using Poetry:

```bash
poetry install
```

This installs all packages defined in `pyproject.toml`.

---

## 3️⃣ Run the FastAPI Application

Start the API server:

```bash
poetry run uvicorn app.main:app --reload
```

If successful, the API will be available at:

```
http://localhost:8000/docs
```

or

```
http://127.0.0.1:8000/docs
```

---

## 4️⃣ Start the Celery Worker

Open **another terminal** (while the API is running):

```bash
cd ai-service
```

Start the Celery worker:

```bash
poetry run python -m celery -A app.core.celery_app.celery_app worker --loglevel=info --pool=solo
```

---

## 🔍 Monitoring Task Execution

When API requests trigger background jobs:

* The **Celery worker console** will display detailed execution stages.
* Logs include processing steps, retries, and task lifecycle events.

---

## 🌸 Flower Dashboard (Optional)

Flower provides a web UI for Celery monitoring.

Access via:

```
http://localhost:5555
```

---

## 🧠 Development Workflow Summary

```
1. docker compose up -d        (infra)
2. poetry install              (ai-service)
3. run FastAPI server
4. start Celery worker
5. open /docs endpoint
```

---

## 🛑 Stopping Services

Stop infrastructure:

```bash
cd infra
docker compose down
```

---

## 🧩 Troubleshooting

### Environment Variables Not Loading

Ensure:

* Both `.env` files exist
* Required values are filled
* Containers restarted after changes

---

### Poetry Command Not Found

Ensure Poetry is added to system PATH:

```bash
poetry --version
```

---

### Celery Cannot Connect to Redis

Check:

* Redis container is running
* `REDIS_PASSWORD` matches in both environments

---

### Database Connection Issues

Verify:

* PostgreSQL container running
* Correct `POSTGRES_PASSWORD`
* `DATABASE_URL` matches configuration

---

## 📘 Notes for Developers

* Infrastructure runs entirely via Docker.
* Application runs locally through Poetry virtual environment.
* Celery executes asynchronous AI processing workflows.
* Worker logs provide deep visibility into request processing stages.

---

## ✅ Pipeline Stages (High Level)

Client Request
      
1. Rate Limit Check
      
2. Input Validation
      
3. Cache Check (Semantic)
      
4. Intent Analysis (Llama3 -> Heuristic Fallback)
      
5. Context Building
      
6. Architecture Generation (Llama3 -> Heuristic Fallback)
      
7. Layout Generation (Llama3 -> Heuristic Fallback)
      
8. Blockly Generation (Llama3 -> Heuristic Fallback)
      
9. Cache Save
      
9. Database Save

