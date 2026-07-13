# Deployment — public demo on Hugging Face Spaces (free CPU tier)

The Docker image from this repo deploys as-is to a **Docker Space**. The
free tier's ephemeral storage is a feature for a demo: every Space restart
is a data wipe (the in-app 24h wipe thread covers long uptimes).

## 1. Create the Space

1. <https://huggingface.co/new-space> → SDK: **Docker** → CPU basic (free).
2. Push this repository to the Space (or add the Space as a second git
   remote and push a `deploy` branch):

   ```bash
   git remote add space https://huggingface.co/spaces/<you>/bone-fracture-demo
   git push space main
   ```

3. Spaces builds straight from the `Dockerfile`. Add this front-matter to
   the **Space's** `README.md` (Spaces reads it for configuration; the
   GitHub README doesn't need it):

   ```yaml
   ---
   title: Bone Fracture Detection
   emoji: 🦴
   colorFrom: blue
   colorTo: gray
   sdk: docker
   app_port: 7860
   ---
   ```

## 2. Space variables & secrets

| Name | Type | Value |
|---|---|---|
| `SECRET_KEY` | secret | any long random string |
| `DEMO_MODE` | variable | `1` |
| `PORT` | variable | `7860` (Spaces' expected port; the image reads `$PORT`) |
| `SECURE_COOKIES` | variable | `1` (Spaces serves HTTPS) |

Do **not** set `CLINICAL_KEY` / `ADMIN_KEY` — demo mode disables clinical
registration entirely and seeds `demo-patient` / `demo-doctor` /
`demo-admin` accounts whose rotating passwords are shown on the login page.

## 3. What DEMO_MODE=1 switches on

- Seeded demo accounts (fresh random passwords each boot, shown at login).
- Clinical/admin registration keys disabled — nobody mints real roles.
- "Try a sample X-ray" gallery on the upload page (clear fracture, subtle
  fracture in the abstention band, two normals, and one deliberately
  non-X-ray image that demos the OOD gate).
- "Public demo — do not upload real patient images" banner on every page.
- Aggressive limits: uploads 3/minute/IP, 5 MB cap.
- 24-hour wipe of scans, jobs, annotations and media (plus ephemeral
  storage on the free tier).

## 4. Resources

The 22 MB Keras models on TF-CPU fit the free tier. Gunicorn runs
`-w 1 --threads 4`; `PRELOAD_MODELS=1` (set in the image) warms the models
off the request path, and the async pipeline's "queued" state hides any
first-request latency.

## Alternatives

Render / Railway free tiers run the same image (set the same env vars);
they sleep aggressively on free plans, which makes the first request slow —
Spaces is the recommended default.
