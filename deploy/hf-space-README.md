---
title: Bone Fracture Detection
emoji: 🦴
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🦴 Bone Fracture Detection — Live Demo

A calibrated two-stage CNN pipeline that screens bone X-rays for fractures,
classifies the fracture type, refuses non-radiographs, and explains its
predictions with an interactive Grad-CAM viewer — in a role-based clinical
portal (Patient / Doctor / Admin).

**This is a public demo.** Do not upload real patient images; data is wiped
periodically. Seeded accounts (`demo-patient`, `demo-doctor`, `demo-admin`)
are shown on the login page, or click **"Try a sample X-ray"** on the upload
page to run the full pipeline in one click.

> ⚕️ Academic project — **not a medical device.** Predictions are for
> educational demonstration only and must never replace evaluation by a
> qualified radiologist or physician.

Source code, honest evaluation, and the train/test-leakage investigation:
**https://github.com/anandjonnadula/Bone-Fracture-Detection-Using-CNN**

<!--
This README configures the Hugging Face Space (the front-matter above).
It intentionally differs from the GitHub README so the GitHub page stays clean.
See docs/DEPLOY.md in the source repo for how to deploy.
-->
