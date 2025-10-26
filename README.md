# Smart Health Assistant — Render deployment

This repository contains a small Flask app that uses a trained model to provide simple symptom-based suggestions.

What I added for Render
- `Procfile` — web: gunicorn app:app --bind 0.0.0.0:$PORT
- `requirements.txt` updated with `gunicorn`
- `runtime.txt` pinned to `python-3.11.6`
- `render.yaml` (optional) describing the web service and a 1GB persistent disk mounted at `/data`

Model files
- The app will look for model files in this order:
  1. Path given by env var `MODEL_PATH` and `VECTORIZER_PATH` (if set).
  2. `/data/models/model.joblib` and `/data/models/vectorizer.joblib` (persistent disk mounted by `render.yaml`).
  3. `model.joblib` and `vectorizer.joblib` in the repository root.

If none are found the app will raise an error on startup. This lets you keep large model files on Render's persistent disk and update them without changing the repo.

Deploying to Render (quick steps)
1. Push your repo to GitHub.
2. On Render.com create a new Web Service and connect the GitHub repo. You can import `render.yaml` via the Dashboard's "Create from YAML" option or create the service manually.
   - Branch: `main`
   - Build Command: `pip install -r requirements.txt` (or leave blank — Render will run pip automatically)
   - Start Command: leave blank (Render will use the `Procfile`) or set `gunicorn app:app --bind 0.0.0.0:$PORT`
3. Ensure the service has the disk attached if you used `render.yaml` (the YAML sets a 1GB disk at `/data`).

Placing model files onto the persistent disk (`/data/models`)
Option A — keep models in the repo (simple)
- Commit `model.joblib` and `vectorizer.joblib` to the repo root and push. Render will build and deploy them with the service.

Option B — upload models to the service disk (recommended for large models)
1. Upload your model files to a publicly accessible URL (S3, Google Cloud Storage, or a temporary file host).
2. Open the Render dashboard, go to your Service -> Shell (this opens a shell inside the running container).
3. Run:

```bash
mkdir -p /data/models
curl -o /data/models/model.joblib "https://example.com/path/to/model.joblib"
curl -o /data/models/vectorizer.joblib "https://example.com/path/to/vectorizer.joblib"
```

4. Restart your service.

Option C — set env vars to point at files you placed elsewhere
- Set `MODEL_PATH` and `VECTORIZER_PATH` environment variables in the Render dashboard to the full filesystem path where your service can read the files.

Notes
- If you keep models in the repo, updating them requires a git push and redeploy.
- Using the persistent disk allows replacing models without redeploying the code.
- Avoid committing very large model files to the repo if you can use the disk or cloud storage instead.

Example: set `MODEL_PATH` via the Render Dashboard env var to `/data/models/model.joblib` after uploading files to `/data/models`.

If you'd like, I can:
- Add a small startup script that downloads models from a protected S3 bucket using credentials stored as Render env vars.
- Add a healthcheck endpoint `/health` that returns 200 OK (I can add this to `app.py`).

---
Small support: to test locally, create a virtualenv, install `-r requirements.txt`, and run `python app.py` (local run uses Flask dev server). For production and Render use gunicorn as configured.
