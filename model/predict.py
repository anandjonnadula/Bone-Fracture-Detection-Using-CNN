import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import cv2
import json
from tensorflow.keras.models import Model
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from datetime import datetime
import matplotlib.pyplot as plt

# Load Class Labels dynamically if available
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
default_labels = [
    'avulsion', 'comminuted', 'compression', 'dislocation', 'greenstick', 'hairline', 
    'impacted', 'intra_articular', 'longitudinal', 'oblique', 'pathological', 'spiral'
]

indices_path = os.path.join(BASE_DIR, "class_indices.json")
if os.path.exists(indices_path):
    try:
        with open(indices_path, "r") as f:
            indices_dict = json.load(f)
            # Sort by index value to get alphabetical labels
            class_labels = [k for k, v in sorted(indices_dict.items(), key=lambda item: item[1])]
            # Clean folder names like 'Avulsion fracture' to 'avulsion'
            class_labels = [label.lower().replace(" fracture", "").replace("-crush", "").replace("fracture ", "").replace(" ", "_") for label in class_labels]
    except Exception as e:
        print("Error loading dynamic class indices:", e)
        class_labels = default_labels
else:
    class_labels = default_labels

# Severity mapping
severity_dict = {
    "hairline": "Mild",
    "greenstick": "Mild",
    "avulsion": "Moderate",
    "spiral": "Moderate",
    "oblique": "Moderate",
    "compression": "Moderate",
    "impacted": "Moderate",
    "pathological": "Severe",
    "dislocation": "Severe",
    "longitudinal": "Severe",
    "intra_articular": "Severe",
    "comminuted": "Severe"
}

# Doctor Recommendation 🧾
recommendation_dict = {
    "Mild": "Rest the affected area and avoid strain. Follow basic care and monitor symptoms.",
    "Moderate": "Consult an orthopedic specialist. Immobilization or minor treatment may be required.",
    "Severe": "Immediate medical attention is required. Possible surgery or advanced treatment needed."
}

# Load Models
stage1_model_path = os.path.join(BASE_DIR, "saved_model", "stage1_model.h5")
stage2_model_path = os.path.join(BASE_DIR, "saved_model", "stage2_model.h5")

stage1_model = tf.keras.models.load_model(stage1_model_path)
stage2_model = tf.keras.models.load_model(stage2_model_path)

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Grad-CAM generation (enhanced for Transfer Learning 🔥)
# -----------------------------
def generate_gradcam(img_path, model):
    try:
        img = preprocess(img_path)
        
        # Identify last conv layer dynamically
        if "mobilenetv2" in model.layers[0].name.lower():
            base_model = model.layers[0]
            last_conv_layer = base_model.get_layer("Conv_1")
            grad_model = Model(
                [model.inputs],
                [last_conv_layer.output, model.output]
            )
        else:
            # Fallback for simple CNN
            grad_model = Model(
                [model.inputs],
                [model.get_layer("conv2d_2").output, model.output]
            )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Original image
        img_orig = cv2.imread(img_path)
        img_orig = cv2.resize(img_orig, (224, 224))

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)

        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay
        superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap_color, 0.4, 0)

        # 🔥 ADD LABEL ON HEATMAP
        cv2.putText(
            superimposed_img,
            "Fracture Area",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),  # Red color
            2,
            cv2.LINE_AA
        )

        gradcam_path = img_path.replace(".jpg", "_gradcam.jpg")
        cv2.imwrite(gradcam_path, superimposed_img)

        return gradcam_path

    except Exception as e:
        print("Grad-CAM generation failed:", e)
        return None



# -----------------------------
# Predict function
# -----------------------------
def predict_fracture(img_path):
    img = preprocess(img_path)

    # Stage 1: Fracture / No Fracture
    stage1_pred = stage1_model.predict(img)[0][0]

    # Dataset folders alphabetically: 'fracture' = class 0, 'no_fracture' = class 1
    # Sigmoid output < 0.5  → closer to class 0 → Fracture Detected
    # Sigmoid output >= 0.5 → closer to class 1 → No Fracture Detected
    if stage1_pred < 0.5:
        result_text = "Fracture Detected"
        # Stage 2: Fracture type
        stage2_pred = stage2_model.predict(img)[0]
        class_index = int(np.argmax(stage2_pred))
        fracture_type = class_labels[class_index]
        confidence = stage2_pred[class_index] * 100

        # Severity
        severity = severity_dict.get(fracture_type.lower(), "Unknown")

        # Recommendation
        recommendation = recommendation_dict.get(severity, "Consult a medical professional.")

        # Grad-CAM image
        gradcam_path = generate_gradcam(img_path, stage2_model)

        return {
            "result": f"Fracture Detected: {fracture_type.capitalize()}",
            "confidence": round(confidence, 2),
            "gradcam": gradcam_path,
            "severity": severity,
            "recommendation": recommendation
        }
    else:
        result_text = "No Fracture Detected"
        confidence = round(stage1_pred * 100, 2)
        return {
            "result": result_text,
            "confidence": confidence,
            "gradcam": None,
            "severity": "None",
            "recommendation": "No medical action required at this time based on the scan."
        }

# -----------------------------
# PDF generation
# -----------------------------
def generate_pdf(report_path, original_img_path, gradcam_img_path, result, confidence, severity, logo_path=None):
    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4
    margin = 50
    current_height = height - margin

    # Logo
    if logo_path and os.path.exists(logo_path):
        c.drawImage(logo_path, margin, current_height-80, width=100, height=50)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin + 120, current_height-50, "Bone Fracture Detection Report")

    current_height -= 90
    c.setFont("Helvetica", 12)
    c.drawString(margin, current_height, f"Report Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    current_height -= 20
    c.setStrokeColor(colors.grey)
    c.setLineWidth(1)
    c.line(margin, current_height, width-margin, current_height)

    # Original Image
    current_height -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, current_height, "Original X-ray Image:")
    current_height -= 10
    if os.path.exists(original_img_path):
        c.drawImage(original_img_path, margin, current_height-250, width=250, height=250, preserveAspectRatio=True)
        c.rect(margin-2, current_height-252, 254, 254, stroke=1, fill=0)

    # Grad-CAM
    if gradcam_img_path and os.path.exists(gradcam_img_path):
        c.drawString(margin + 280, current_height, "Grad-CAM Highlight:")
        c.drawImage(gradcam_img_path, margin + 280, current_height-250, width=250, height=250, preserveAspectRatio=True)
        c.rect(margin+278, current_height-252, 254, 254, stroke=1, fill=0)

    current_height -= 270

    # Summary Table
    data = [["Prediction", "Confidence", "Severity"]]
    data.append([result, f"{confidence}%", severity])
    table = Table(data, colWidths=[200, 100, 100])

    severity_color = colors.green if severity=="Mild" else (colors.orange if severity=="Moderate" else colors.red)
    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('TEXTCOLOR', (2,1), (2,1), severity_color)
    ])
    table.setStyle(style)
    table.wrapOn(c, width, height)
    table.drawOn(c, margin, current_height-50)

    c.showPage()
    c.save()
    
# -----------------------------
# Doctor Recommendation 🧾
# -----------------------------
recommendation_dict = {
    "Mild": "Rest the affected area and avoid strain. Follow basic care and monitor symptoms.",
    "Moderate": "Consult an orthopedic specialist. Immobilization or minor treatment may be required.",
    "Severe": "Immediate medical attention is required. Possible surgery or advanced treatment needed."
}

# -----------------------------
# Model Performance Data 📊
# -----------------------------
def get_model_performance():
    # Dummy values (replace with real history if available)
    return {
        "epochs": list(range(1, 11)),
        "accuracy": [0.72, 0.78, 0.82, 0.86, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95],
        "val_accuracy": [0.70, 0.75, 0.80, 0.84, 0.86, 0.88, 0.89, 0.90, 0.91, 0.92]
    }