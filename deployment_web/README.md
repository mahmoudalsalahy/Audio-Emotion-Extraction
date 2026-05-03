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

## Browser-Only GitHub Pages Predictions

The GitHub Pages frontend can run predictions without a backend using the
exported browser model at:

```text
deployment_web/frontend/public/model/browser-model.json
```

The browser version uses the RandomForest and KNN components from the saved
stacking model and a JavaScript implementation of the audio feature extraction pipeline.
It avoids FastAPI hosting entirely, so no `VITE_API_BASE` variable is required
for GitHub Pages.

To regenerate the browser model after retraining:

```powershell
python scripts/export_browser_model.py
```

Then rebuild the frontend:

```powershell
cd deployment_web/frontend
cmd /c npm run build
```

## Optional Hosted Backend

The FastAPI backend is still available if you want server-side predictions
instead of the browser model.

If Render asks for payment, use Hugging Face Spaces instead. The backend folder
includes Docker deployment files that work with Hugging Face Spaces:

- `deployment_web/backend/Dockerfile`
- `deployment_web/backend/README.md`
- `deployment_web/backend/models/.gitkeep`
- `deployment_web/backend/data/.gitkeep`

Create a new Hugging Face Space:

1. Choose `Docker` as the Space SDK.
2. Upload the contents of `deployment_web/backend` to the Space repository.
3. Upload `model-v1.pkl` to `models/model-v1.pkl`.
4. Optionally upload `xai-data.pkl` to `data/xai-data.pkl`.
5. Wait for the Space to build and start.

Your backend URL will look like:

```text
https://your-username-audio-emotion-api.hf.space
```

Then add that URL to GitHub Actions variables:

1. Open the GitHub repository.
2. Go to `Settings > Secrets and variables > Actions > Variables`.
3. Add `VITE_API_BASE` with the Hugging Face Space URL.
4. Re-run the `Deploy frontend to GitHub Pages` workflow.

The backend needs `models/model-v1.pkl` to return predictions. Without the
model, `/health` will return `model_missing`.

Render is still supported if you decide to use it later. This repository also
includes a `render.yaml` blueprint for Render:

1. Push this repository to GitHub.
2. Open Render and create a new `Blueprint` from the repository.
3. Render will detect `render.yaml` and create `audio-emotion-api`.
4. After deployment, copy the backend URL, for example:

```text
https://audio-emotion-api.onrender.com
```

For Render, provide the model file on the backend host or set `MODEL_PATH` to
the hosted file path.
