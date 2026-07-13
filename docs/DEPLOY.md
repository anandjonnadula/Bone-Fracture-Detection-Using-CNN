# Deploy the live demo

> **Heads-up (mid-2026):** Hugging Face moved the **Docker and Gradio** Space
> SDKs behind a paid **PRO** subscription — only **Static** Spaces are free now,
> and those can't run a Python/TensorFlow backend. So the recommended free host
> is **Google Cloud Run** (below). The old Hugging Face steps are kept at the
> bottom for anyone who has HF PRO.

---

## Google Cloud Run (recommended, free)

Cloud Run runs this repo's Docker image as-is, **scales to zero** when idle
(so a low-traffic resume demo stays within the free tier ≈ $0), and gives a
professional URL like `https://bone-fracture-detection-xxxx.run.app`.

You need a Google account with **billing enabled** — a card is required for
verification, but you stay inside the free tier and set a low instance cap so
you aren't charged. (The only tiny non-zero cost is ~a few cents/month to store
the ~3 GB container image; negligible, and you can delete old revisions.)

### One key setting

Cloud Run allocates CPU **only during a request**, so the app must run
inference *in* the request rather than in a background thread. Pass
**`SYNC_JOBS=1`** (shown below) — the upload then processes synchronously and
redirects straight to the result. Everything else is unchanged.

### Steps (all in the browser — no local installs)

1. **Create a project & enable billing**
   - Sign in at <https://console.cloud.google.com> → create a project
     (e.g. `bone-fracture-demo`).
   - Billing → link a billing account (adds a card; free-tier usage isn't
     charged). Optionally set a **Budget alert** at $1 for peace of mind.

2. **Open Cloud Shell** (the `>_` icon, top-right of the console). It's a free
   browser terminal with `gcloud`, Docker and git preinstalled.

3. **Clone and deploy** — paste this (replace `<KEY>` with your SECRET_KEY):

   ```bash
   git clone https://github.com/anandjonnadula/Bone-Fracture-Detection-Using-CNN.git
   cd Bone-Fracture-Detection-Using-CNN

   gcloud run deploy bone-fracture-detection \
     --source . \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 1 \
     --timeout 300 \
     --max-instances 3 \
     --set-env-vars SECRET_KEY=<KEY>,DEMO_MODE=1,SECURE_COOKIES=1,SYNC_JOBS=1,PRELOAD_MODELS=0
   ```

   - When prompted, let it **enable the required APIs** (Run, Cloud Build,
     Artifact Registry) and pick region `us-central1` if asked.
   - It uploads the source, builds the image with Cloud Build (installs
     TensorFlow — **~5–10 min the first time**), and deploys.
   - `2Gi` memory is required (TensorFlow needs ~1.5 GB). `--max-instances 3`
     caps cost. `--allow-unauthenticated` makes it public.

4. **Open the URL** it prints (`Service URL: https://…run.app`). The login page
   lists the demo accounts; the upload page has "Try a sample X-ray."

   > First request loads the models (~30–60 s) and the whole pipeline runs
   > during that upload request, so the very first scan is slow; later scans
   > take a few seconds. (With `SYNC_JOBS=1` the progress stepper doesn't
   > animate — the result appears when processing finishes. That's expected on
   > the free, CPU-throttled tier.)

### Update the demo later

```bash
cd Bone-Fracture-Detection-Using-CNN && git pull
gcloud run deploy bone-fracture-detection --source . --region us-central1 \
  --set-env-vars SECRET_KEY=<KEY>,DEMO_MODE=1,SECURE_COOKIES=1,SYNC_JOBS=1,PRELOAD_MODELS=0
```

### Staying free / cost control

- Cloud Run free tier (per month): 2M requests, 360k GiB-seconds, 180k
  vCPU-seconds — a personal demo won't come close. Idle = $0 (scales to zero).
- Keep `--max-instances` small (3) and set a **Budget alert**.
- To stop billing entirely later: `gcloud run services delete bone-fracture-detection --region us-central1`
  and delete the Artifact Registry image.

### Put it on your resume

- **Live demo:** the `https://…run.app` URL Cloud Run printed
- **Source:** `https://github.com/anandjonnadula/Bone-Fracture-Detection-Using-CNN`

Add the live URL to the top of the GitHub `README.md` (there's a placeholder).

### Troubleshooting

- **Build times out / fails** — re-run the command; Cloud Build caches layers so
  the second build is faster. TensorFlow is the slow part.
- **Container fails to start** — check *Cloud Run → Logs*. Ensure `SYNC_JOBS=1`
  and `--memory 2Gi` are set (out-of-memory shows as the container being killed).
- **502/timeout on first scan** — the first request loads models; `--timeout 300`
  gives it room. Refresh and try a sample again once it's warm.

---

## Hugging Face Spaces (only if you have HF PRO)

If you have (or get) HF PRO, a Docker Space also works. Create a **Docker**
Space (CPU basic), then in *Settings → Variables and secrets* set
`SECRET_KEY` (secret), `DEMO_MODE=1`, `SECURE_COOKIES=1`. The image defaults to
port 7860 to match. Push a deploy branch (git-LFS for the `.keras` models):

```bash
git lfs install
git checkout -b hf-deploy
git lfs track "*.keras" "*.npz"
cp deploy/hf-space-README.md README.md
git add .gitattributes README.md
git add --renormalize model/saved_model
git commit -m "Configure Hugging Face Space"
git remote add space https://huggingface.co/spaces/<USER>/bone-fracture-detection
git push space hf-deploy:main          # username = HF user, password = HF write token
git checkout main
```

## Other hosts

Render / Railway / Fly.io **free** tiers give only 512 MB RAM and cannot load
TensorFlow. Their paid tiers work with the same image (set `PORT`, `SECRET_KEY`,
`DEMO_MODE=1`, `SYNC_JOBS=1`). Oracle Cloud "Always Free" ARM VMs (up to 24 GB
RAM) can run the container for free too, but require provisioning and securing a
VM yourself — more work than Cloud Run.
