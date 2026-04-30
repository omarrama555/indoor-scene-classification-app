import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

# --- 1. إعدادات الصفحة والـ CSS المخصص ---
st.set_page_config(page_title="Pro Indoor AI Explorer", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .prediction-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. قائمة الفئات (67 فئة) ---
CLASSES = ['airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar', 'bathroom', 'bedroom', 'bookstore', 'bowling', 'buffet', 'casino', 'children_room', 'church_inside', 'classroom', 'cloister', 'closet', 'clothingstore', 'computerroom', 'concert_hall', 'corridor', 'deli', 'dentaloffice', 'dining_room', 'elevator', 'fastfood_restaurant', 'florist', 'gameroom', 'garage', 'greenhouse', 'grocerystore', 'gym', 'hairsalon', 'hospitalroom', 'inside_bus', 'inside_subway', 'jewelleryshop', 'kindergarden', 'kitchen', 'laboratorywet', 'laundromat', 'library', 'livingroom', 'lobby', 'locker_room', 'mall', 'meeting_room', 'movietheater', 'museum', 'nursery', 'office', 'operating_room', 'pantry', 'poolinside', 'prisoncell', 'restaurant', 'restaurant_kitchen', 'shoeshop', 'stairscase', 'studiomusic', 'subway', 'toystore', 'trainstation', 'tv_studio', 'videostore', 'waitingroom', 'warehouse', 'winecellar']

# --- 3. بناء وتحميل النموذج ---
def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    return model

@st.cache_resource
def load_model():
    model = build_model(len(CLASSES))
    state_dict = torch.load('indoor_model_weights.pth', map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

# --- 4. معالجة الصور والـ Grad-CAM ---
def get_prediction(model, img_pil):
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_tensor = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
    return probs[0]

def generate_gradcam(model, img_pil, target_category):
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    img_np = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    input_tensor = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img_pil.resize((224, 224))).unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization

# --- 5. الواجهة الرئيسية ---
st.sidebar.title("⚙️ Control Panel")
show_history = st.sidebar.checkbox("Show Search History", value=True)
analysis_mode = st.sidebar.selectbox("Analysis Depth", ["Standard", "Deep (Grad-CAM)"])

st.title("🏢 Pro Indoor AI Classifier")
st.markdown("---")

try:
    model = load_model()
    
    uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if 'history' not in st.session_state: st.session_state.history = []
        
        for uploaded_file in uploaded_files:
            col1, col2 = st.columns([1, 1])
            img = Image.open(uploaded_file).convert('RGB')
            
            with col1:
                st.image(img, caption=f"Original: {uploaded_file.name}", use_container_width=True)
            
            probs = get_prediction(model, img)
            top5_conf, top5_labels = torch.topk(probs, 5)
            
            with col2:
                st.subheader("📊 Analysis Results")
                main_pred = CLASSES[top5_labels[0]]
                main_conf = top5_conf[0].item()
                
                st.metric("Top Prediction", main_pred, f"{main_conf*100:.2f}%")
                
                # Top-5 Chart
                chart_data = pd.DataFrame({
                    'Class': [CLASSES[i] for i in top5_labels],
                    'Confidence': top5_conf.numpy() * 100
                })
                st.bar_chart(chart_data.set_index('Class'))
                
                if analysis_mode == "Deep (Grad-CAM)":
                    with st.spinner('Generating Heatmap...'):
                        heatmap = generate_gradcam(model, img, top5_labels[0].item())
                        st.image(heatmap, caption="AI Focus Area (Grad-CAM)", use_container_width=True)

            st.session_state.history.append({"file": uploaded_file.name, "pred": main_pred})
            st.markdown("---")

        # Export Feature
        if st.button("📩 Export Results as Text"):
            report = "\n".join([f"{h['file']}: {h['pred']}" for h in st.session_state.history])
            st.download_button("Download Report", report, file_name="ai_report.txt")

except Exception as e:
    st.error(f"System Error: {e}")
