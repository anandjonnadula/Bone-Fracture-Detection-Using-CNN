# Deploy the live demo — Hugging Face Spaces (free)

Target: a **Docker Space** on Hugging Face — free CPU tier with 16 GB RAM
(enough for TensorFlow; Render/Railway free tiers only give 512 MB and can't
run TF). The image from this repo's `Dockerfile` runs there as-is: it listens
on port **7860** and creates its SQLite DB + media dir in world-writable paths,
so it works whether the platform runs the container as root or a non-root user.

You need: a free Hugging Face account and a Hugging Face **write access token**.
The steps below are the parts only you can do (they use your account); the repo
is already prepared for all of it.

---

## 1. Create the Space

1. Sign in at <https://huggingface.co> (free — <https://huggingface.co/join>).
2. Go to <https://huggingface.co/new-space>.
3. Fill in:
   - **Owner**: your username
   - **Space name**: `bone-fracture-detection` (becomes the URL)
   - **License**: MIT
   - **SDK**: **Docker** → **Blank**
   - **Hardware**: **CPU basic** (free)
   - **Visibility**: **Public** (so you can share it on your resume)
4. Click **Create Space**. It starts empty.

## 2. Add the Space's secrets & variables

In the Space: **Settings → Variables and secrets → New secret / New variable**:

| Name | Kind | Value |
|---|---|---|
| `SECRET_KEY` | **secret** | the 64-char key generated for you (or any long random string) |
| `DEMO_MODE` | variable | `1` |
| `SECURE_COOKIES` | variable | `1` |

Do **not** set `CLINICAL_KEY` / `ADMIN_KEY` — demo mode disables clinical
registration and seeds `demo-patient` / `demo-doctor` / `demo-admin` accounts
(their rotating passwords appear on the login page). `PORT` is not needed — the
image already defaults to 7860, which the Space README's `app_port` matches.

## 3. Get a write token

<https://huggingface.co/settings/tokens> → **New token** → type **Write** →
copy it. You'll paste it as the *password* when git pushes to the Space.

## 4. Push the code to the Space

The Space is its own git repo. Push a **deploy branch** to it — this keeps your
GitHub `main` untouched (GitHub keeps the clean README and regular-git models;
the Space gets the front-matter README and LFS-tracked models). Hugging Face
requires **git-LFS** for files > 10 MB (the two `.keras` models), so:

```bash
# Install git-LFS once if you don't have it: https://git-lfs.com
git lfs install

# From the project folder, create a deploy branch
git checkout -b hf-deploy

# Track the large model files with LFS and use the Space's README (front-matter)
git lfs track "*.keras" "*.npz"
cp deploy/hf-space-README.md README.md
git add .gitattributes README.md
git add --renormalize model/saved_model        # convert the committed models to LFS pointers
git commit -m "Configure Hugging Face Space (LFS models + Space README)"

# Add the Space as a remote (replace <USER>) and push this branch to its main
git remote add space https://huggingface.co/spaces/<USER>/bone-fracture-detection
git push space hf-deploy:main
#   Username: <your HF username>
#   Password: <your HF write token from step 3>

# Go back to your normal branch — GitHub main is unchanged
git checkout main
```

## 5. Watch it build

The Space's **Logs / App** tab shows the Docker build. First build takes
**~10–15 min** (it installs TensorFlow). When it says *Running*, open the Space
URL: `https://huggingface.co/spaces/<USER>/bone-fracture-detection`.

Try it: the login page lists the demo accounts, and the upload page has a
**"Try a sample X-ray"** strip (including a non-X-ray that demos the OOD gate).

## 6. Put it on your resume

- **Live demo:** `https://huggingface.co/spaces/<USER>/bone-fracture-detection`
- **Source:** `https://github.com/anandjonnadula/Bone-Fracture-Detection-Using-CNN`

Add the live link to the top of the GitHub `README.md` (there's a placeholder
for it) so visitors can jump straight to the demo.

---

## Updating the demo later

After changes on `main`:

```bash
git checkout hf-deploy
git merge main -m "sync"          # bring in the changes
git checkout main -- README.md && cp deploy/hf-space-README.md README.md
git add README.md && git commit -m "keep Space README"   # if README changed
git push space hf-deploy:main
git checkout main
```

## Troubleshooting

- **Build fails at `pip install`** — confirm the Space is a *Docker* SDK Space
  (not Gradio/Streamlit); it must build from the `Dockerfile`.
- **App builds but won't start / 503** — check the Space *Logs*. The first model
  load takes ~30–60 s; the async pipeline hides this behind the "queued" state.
- **Push rejected for a large file** — you skipped `git lfs track` / the
  `--renormalize` step; redo step 4 on a fresh `hf-deploy` branch.
- **Out of memory** — the free CPU tier (16 GB) is plenty; if you switched to a
  smaller paid tier, go back to CPU basic.

## Alternatives (not recommended for this app)

Render / Railway free tiers (512 MB RAM) cannot load TensorFlow. Their paid
tiers work with the same image (set `PORT`, `SECRET_KEY`, `DEMO_MODE=1`), but
Hugging Face's free CPU tier is the better fit for an ML demo.
