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

## GitHub Pages Deployment

This repo includes a GitHub Actions workflow at
`.github/workflows/deploy-frontend.yml` that builds `deployment_web/frontend`
and publishes the generated `dist` folder to GitHub Pages.

1. Push the repository to GitHub on the `main` branch.
2. In GitHub, open `Settings > Pages`.
3. Set `Build and deployment > Source` to `GitHub Actions`.
4. Push a new commit or run the `Deploy frontend to GitHub Pages` workflow manually.

GitHub Pages only hosts the React frontend. The FastAPI backend must be hosted
separately if you want live predictions online. After deploying the backend,
add a repository variable named `VITE_API_BASE` in
`Settings > Secrets and variables > Actions > Variables`, for example:

```text
https://your-api-host.example.com
```

Leave `VITE_API_BASE` empty for local development, where Vite proxies `/api`
to `http://127.0.0.1:8000`.

On the backend host, set `CORS_ORIGINS` to your GitHub Pages site URL, for
example:

```text
https://your-github-username.github.io
```

If you use multiple frontend origins, separate them with commas.
