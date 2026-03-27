# Medical Image Anomaly Detector (MRI Tumor Classification)

This project trains an EfficientNetB0-based classifier on the Brain Tumor MRI dataset and provides a Flask web interface to:
- Upload MRI images
- Predict tumor class (`glioma`, `meningioma`, `pituitary`, `notumor`)
- Visualize region of interest with Grad-CAM

## Project Structure

- `Dataset/Training` and `Dataset/Testing`: local dataset folders
- `train_model.py`: model training + evaluation + artifact generation
- `app.py`: Flask inference app
- `app/templates`: HTML pages
- `app/static/css/style.css`: UI styling
- `models/`: trained model and class map
- `results/`: Grad-CAM outputs and plots
- `uploads/`: uploaded images during inference

## 1) Install dependencies

Use Python 3.10 or 3.11 for best TensorFlow compatibility.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Train model

```bash
python train_model.py
```

Training outputs:
- `models/brain_tumor_efficientnetb0.keras`
- `models/class_indices.json`
- `results/accuracy_curve.png`
- `results/loss_curve.png`
- `results/confusion_matrix.png`
- `results/classification_report.txt`

## 3) Run web app

```bash
python app.py
```

Open in browser:
- `http://127.0.0.1:5000`

## One-click demo (train if needed + launch app)

```bash
chmod +x run_demo.sh
./run_demo.sh
```

This script will:
- create `.venv` if missing,
- install dependencies,
- train the model only if artifacts are missing,
- start the Flask app.

## Deploy (Render)

This repo is now deployment-ready with:
- `Procfile` (Gunicorn start command)
- `render.yaml` (Render service config)
- `/health` endpoint in app for health checks

### Steps

1. Push this project to GitHub.
2. In Render, choose **New +** → **Blueprint**.
3. Connect your GitHub repo and deploy.
4. Ensure the trained model artifacts are in the repo:
	- `models/brain_tumor_efficientnetb0.keras`
	- `models/class_indices.json`

Render will install from `requirements.txt` and run:

```bash
gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT
```

## Hackathon Demo Tips

- Keep `results/confusion_matrix.png` and `results/classification_report.txt` ready for judges.
- Mention class balancing + augmentation + transfer learning + Grad-CAM explainability.
- Show one known tumor image and one non-tumor image in live demo.
