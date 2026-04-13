import streamlit as st
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from groq import Groq
import json
import os
from dotenv import load_dotenv

# ── LOAD ENV ─────────────────────────────────────────────

load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── PAGE CONFIG ──────────────────────────────────────────

st.set_page_config(
page_title="🌿 Plant Disease Detector",
layout="wide",
initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────

st.markdown("""

<style>
.main { background-color: #0E1117; }

h1 { font-size: 2.5rem !important; font-weight: 700 !important; }

.card {
    background: #161B22;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 10px;
}

.treatment-box {
    background: #0B3D2E;
    padding: 12px;
    border-radius: 10px;
    border-left: 4px solid #00FFAA;
    margin-bottom: 10px;
}

.stProgress > div > div {
    height: 10px;
    border-radius: 10px;
}
</style>

""", unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────────────────────────

@st.cache_resource
def load_model():
with open("class_names.json") as f:
class_names = json.load(f)

```
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Sequential(
        nn.Identity(),
        nn.Linear(model.last_channel, len(class_names))
    )
)

model.load_state_dict(torch.load("mobilenetv2_scratch.pth", map_location="cpu"))
model.eval()

cam = GradCAM(model=model, target_layers=[model.features[18][0]])

return model, cam, class_names
```

model, cam, class_names = load_model()

# ── TRANSFORM ────────────────────────────────────────────

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])
])

# ── FUNCTIONS ─────────────────────────────────────────────

def get_severity(conf):
if conf < 0.60: return "Low", "🟢"
elif conf <= 0.80: return "Medium", "🟡"
else: return "High", "🔴"

def get_treatment(crop, disease):
if not os.environ.get("GROQ_API_KEY"):
return "⚠️ No API key found. Add GROQ_API_KEY in .env"

```
try:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[
            {"role": "system", "content": """You are an agricultural expert. Give concise treatment:
```

DIAGNOSIS, IMMEDIATE ACTION, SPRAY/CHEMICAL, PREVENTION."""},
{"role": "user", "content": f"Plant: {crop}\nDisease: {disease}"}
]
)
return response.choices[0].message.content
except Exception as e:
return f"Error: {str(e)}"

def predict_all(img):
if img.mode != "RGB":
img = img.convert("RGB")

```
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)

top3_probs, top3_idx = torch.topk(probs, 3, dim=1)

top3 = [
    {
        "class": class_names[top3_idx[0][i].item()],
        "confidence": top3_probs[0][i].item()
    }
    for i in range(3)
]

pred = top3[0]["class"]
conf = top3[0]["confidence"]
severity, icon = get_severity(conf)

parts = pred.split("___")
crop = parts[0].replace("_", " ")
disease = parts[1].replace("_", " ")

treatment = get_treatment(crop, disease)

# GradCAM
cam_map = cam(input_tensor=tensor)[0]
img_np = np.array(img.resize((224,224))).astype(np.float32)/255.0
heatmap = show_cam_on_image(img_np, cam_map, use_rgb=True)

return {
    "crop": crop,
    "disease": disease,
    "confidence": conf,
    "severity": severity,
    "icon": icon,
    "top3": top3,
    "heatmap": Image.fromarray(heatmap),
    "treatment": treatment
}
```

# ── UI ────────────────────────────────────────────────────

st.title("🌿 Plant Disease Detector")
st.caption("AI-powered diagnosis with Grad-CAM explainability")

# Sidebar

st.sidebar.header("📷 Input")
uploaded = st.sidebar.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

if uploaded:
img = Image.open(uploaded)

```
with st.spinner("Analyzing..."):
    res = predict_all(img)

# Images
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🖼️ Visual Analysis</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
c1.image(img, use_container_width=True)
c2.image(res["heatmap"], use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Metrics
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Analysis</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Crop", res["crop"])
m2.metric("Confidence", f"{res['confidence']*100:.1f}%")
m3.metric("Severity", f"{res['icon']} {res['severity']}")

st.markdown('</div>', unsafe_allow_html=True)

# Predictions
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔝 Top Predictions</div>', unsafe_allow_html=True)

for i,p in enumerate(res["top3"]):
    clean = p["class"].replace("___"," — ").replace("_"," ")
    st.write(f"**{i+1}. {clean}** ({p['confidence']*100:.1f}%)")
    st.progress(p["confidence"])

st.markdown('</div>', unsafe_allow_html=True)

# Treatment
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💊 Treatment Plan</div>', unsafe_allow_html=True)

sections = ["DIAGNOSIS","IMMEDIATE ACTION","SPRAY","PREVENTION"]

for sec in sections:
    if sec in res["treatment"]:
        try:
            text = res["treatment"].split(sec+":")[1].split("\n")[0]
            st.markdown(f"**{sec}**")
            st.markdown(f"<div class='treatment-box'>{text}</div>", unsafe_allow_html=True)
        except:
            pass

st.markdown('</div>', unsafe_allow_html=True)
```

else:
st.info("👈 Upload a leaf image to start")
