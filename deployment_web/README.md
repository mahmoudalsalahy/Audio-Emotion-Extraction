# SER Deployment Web App

React 19 frontend plus FastAPI backend for the saved notebook model.

## Required Training Artifact

Run the notebook through Week 6, then run the save-model cell added before the GUI section. The API expects:

- `models/model-v1.pkl`
- `data/xai-data.pkl` optional

## Backend

```powershell
cd deployment_web/backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

## Frontend

```powershell
cd deployment_web/frontend
cmd /c npm install
cmd /c npm run dev
```

Open `http://127.0.0.1:5173`.

The frontend calls the backend through the Vite `/api` proxy, so keep both
servers running at the same time.
