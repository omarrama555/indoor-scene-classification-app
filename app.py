import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import time
import json
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import base64
from io import BytesIO

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Pro Indoor AI Explorer Pro", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")

# --- 2. قائمة الفئات (67 فئة) مع شرح لكل فئة ---
CLASSES_DESCRIPTION = {
    'airport_inside': 'داخل المطار - مناطق الانتظار والصالات',
    'artstudio': 'استوديو فني - مكان للرسم والتصميم',
    'auditorium': 'قاعة محاضرات - مسرح أو قاعة كبيرة',
    'bakery': 'مخبز - مكان لصنع الخبز والحلويات',
    'bar': 'بار - مكان للمشروبات',
    'bathroom': 'حمام - دورة مياه',
    'bedroom': 'غرفة نوم - مكان للنوم',
    'bookstore': 'مكتبة - بيع الكتب',
    'bowling': 'صالة بولينج',
    'buffet': 'بوفيه مفتوح',
    'casino': 'كازينو - قمار',
    'children_room': 'غرفة أطفال',
    'church_inside': 'داخل كنيسة',
    'classroom': 'فصل دراسي',
    'cloister': 'كلوستر - فناء دير',
    'closet': 'خزانة ملابس',
    'clothingstore': 'محل ملابس',
    'computerroom': 'غرفة حواسيب',
    'concert_hall': 'قاعة حفلات',
    'corridor': 'ممر - دهليز',
    'deli': 'دلي - سوبرماركت صغير',
    'dentaloffice': 'عيادة أسنان',
    'dining_room': 'غرفة طعام',
    'elevator': 'مصعد',
    'fastfood_restaurant': 'مطعم وجبات سريعة',
    'florist': 'محل ورد',
    'gameroom': 'غرفة ألعاب',
    'garage': 'كراج - جراج',
    'greenhouse': 'بيت زجاجي - زراعة',
    'grocerystore': 'بقالة',
    'gym': 'نادي رياضي - جيم',
    'hairsalon': 'صالون حلاقة',
    'hospitalroom': 'غرفة مستشفى',
    'inside_bus': 'داخل الباص',
    'inside_subway': 'داخل المترو',
    'jewelleryshop': 'محل مجوهرات',
    'kindergarden': 'روضة أطفال',
    'kitchen': 'مطبخ',
    'laboratorywet': 'معمل رطب - كيمياء',
    'laundromat': 'مغسلة ملابس',
    'library': 'مكتبة عامة',
    'livingroom': 'غرفة معيشة - صالون',
    'lobby': 'لوبي - بهو فندق',
    'locker_room': 'غرفة خزائن - خلع ملابس',
    'mall': 'مول تجاري',
    'meeting_room': 'غرفة اجتماعات',
    'movietheater': 'سينما',
    'museum': 'متحف',
    'nursery': 'حضانة أطفال',
    'office': 'مكتب',
    'operating_room': 'غرفة عمليات',
    'pantry': 'مخزن طعام',
    'poolinside': 'مسبح داخلي',
    'prisoncell': 'زنزانة سجن',
    'restaurant': 'مطعم',
    'restaurant_kitchen': 'مطبخ مطعم',
    'shoeshop': 'محل أحذية',
    'stairscase': 'سلم - درج',
    'studiomusic': 'استوديو موسيقى',
    'subway': 'مترو أنفاق',
    'toystore': 'محل ألعاب',
    'trainstation': 'محطة قطار',
    'tv_studio': 'استوديو تلفزيون',
    'videostore': 'محل فيديو',
    'waitingroom': 'غرفة انتظار',
    'warehouse': 'مستودع - مخزن',
    'winecellar': 'قبو نبيذ'
}

CLASSES = list(CLASSES_DESCRIPTION.keys())

# --- 3. الثيمات الجاهزة ---
THEMES = {
    "Default": {
        "bg": "#f0f2f6",
        "card": "white",
        "text": "black",
        "button": "#ff4b4b"
    },
    "Dark Knight": {
        "bg": "#0e1117",
        "card": "#1e1e2f",
        "text": "white",
        "button": "#ff4b4b"
    },
    "Nature Green": {
        "bg": "#e8f5e9",
        "card": "#c8e6c9",
        "text": "#1b5e20",
        "button": "#4caf50"
    },
    "Ocean Blue": {
        "bg": "#e3f2fd",
        "card": "#bbdefb",
        "text": "#0d47a1",
        "button": "#2196f3"
    },
    "Royal Purple": {
        "bg": "#f3e5f5",
        "card": "#e1bee7",
        "text": "#4a148c",
        "button": "#9c27b0"
    }
}

def apply_theme(theme_name):
    theme = THEMES.get(theme_name, THEMES["Default"])
    st.markdown(f"""
        <style>
        .main {{ background-color: {theme["bg"]}; }}
        .stApp {{ background-color: {theme["bg"]}; }}
        .stButton>button {{ background-color: {theme["button"]} !important; color: white !important; border-radius: 20px !important; }}
        .prediction-card {{ background-color: {theme["card"]}; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: {theme["text"]}; }}
        .stMetric {{ background-color: {theme["card"]}; border-radius: 10px; padding: 10px; }}
        h1, h2, h3, h4, h5, p, span {{ color: {theme["text"]}; }}
        </style>
    """, unsafe_allow_html=True)

# --- 4. بناء وتحميل النموذج ---
def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    return model

@st.cache_resource
def load_model():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Loading model... 0%")
    model = build_model(len(CLASSES))
    progress_bar.progress(20)
    
    status_text.text("📥 Loading weights... 40%")
    try:
        state_dict = torch.load('indoor_model_weights.pth', map_location='cpu')
        progress_bar.progress(70)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        status_text.text("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.warning("⚠️ Model file not found! Using empty model for testing")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    
    progress_bar.progress(100)
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    model.eval()
    return model

# --- 5. معالجة الصور ---
def get_prediction(model, img_pil):
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_tensor = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
    return probs[0]

def generate_gradcam(model, img_pil, target_category):
    if not GRAD_CAM_AVAILABLE:
        return None
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    img_np = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    input_tensor = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img_pil.resize((224, 224))).unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization

# --- 6. دوال الفيتشرات الجديدة ---

def get_image_download_link(img, filename="image.png"):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">Download Image</a>'
    return href

def send_email_report(to_email, subject, content):
    # هاي دالة تجريبية - محتاجة إعدادات SMTP حقيقية
    st.info(f"📧 Mock email sent to {to_email}")
    return True

def save_session(session_data, filename="session.json"):
    with open(filename, 'w') as f:
        json.dump(session_data, f)
    return filename

def load_session(filename="session.json"):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return None

# --- 7. تحميل النموذج ---
try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# --- 8. إدارة الجلسة والمتغيرات ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = {}

# --- 9. الترجمات ---
TEXTS = {
    "English": {
        "title": "🏢 Pro Indoor AI Explorer Pro",
        "welcome": "Welcome to Indoor Scene Classification",
        "upload": "📤 Upload Image(s)",
        "predict": "🔍 Predict",
        "result": "📊 Analysis Results",
        "confidence": "Confidence",
        "processing_time": "Processing time",
        "download": "Download Report",
        "history": "Classification History"
    },
    "العربية": {
        "title": "🏢 المستكشف الاحترافي للأماكن الداخلية",
        "welcome": "مرحباً بك في تصنيف المشاهد الداخلية",
        "upload": "📤 رفع الصور",
        "predict": "🔍 تصنيف",
        "result": "📊 نتائج التحليل",
        "confidence": "نسبة الثقة",
        "processing_time": "وقت المعالجة",
        "download": "تحميل التقرير",
        "history": "سجل التصنيفات"
    }
}

# --- 10. واجهة تسجيل الدخول ---
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Welcome to Pro Indoor AI")
        username_input = st.text_input("👤 Enter your name:")
        lang = st.selectbox("🌍 Language / اللغة", ["English", "العربية"])
        if username_input:
            st.session_state.username = username_input
            st.session_state.language = lang
            st.session_state.logged_in = True
            st.rerun()
    st.stop()

# --- 11. الواجهة الرئيسية ---
lang = st.session_state.language
txt = TEXTS[lang]

# Sidebar
st.sidebar.title(f"👋 {txt['welcome']}, {st.session_state.username}!")
st.sidebar.markdown("---")

# Theme selector
theme_name = st.sidebar.selectbox("🎨 Theme", list(THEMES.keys()))
apply_theme(theme_name)

# Language switcher
new_lang = st.sidebar.selectbox("🌍 Language", ["English", "العربية"], index=0 if lang == "English" else 1)
if new_lang != lang:
    st.session_state.language = new_lang
    st.rerun()

# Analysis settings
analysis_mode = st.sidebar.selectbox("🔬 Analysis Depth", ["Standard", "Deep (Grad-CAM)"])
top_k = st.sidebar.selectbox("📊 Top K Predictions", [3, 5, 10], index=1)

# New Features in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Advanced Features")

# Feature 1: Camera Capture
camera_enabled = st.sidebar.checkbox("📸 Camera Mode")
if camera_enabled:
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        img = Image.open(camera_image).convert('RGB')
        st.session_state.camera_img = img

# Feature 2: Voice input (HTML/JS)
st.sidebar.markdown("🎤 Voice Command")
voice_html = """
<script>
function startRecording() {
    var recognition = new webkitSpeechRecognition();
    recognition.lang = 'ar-EG';
    recognition.onresult = function(event) {
        document.getElementById('voice-result').value = event.results[0][0].transcript;
    }
    recognition.start();
}
</script>
<button onclick="startRecording()">🎤 Speak</button>
<input id="voice-result" placeholder="Voice text will appear here">
"""
st.sidebar.components.html(voice_html, height=100)

# Feature 3: Batch processing
batch_mode = st.sidebar.checkbox("🔄 Batch Processing Mode")
if batch_mode:
    batch_files = st.sidebar.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="batch")

# Feature 4: Save/Load session
st.sidebar.markdown("---")
if st.sidebar.button("💾 Save Session"):
    session_data = {
        "username": st.session_state.username,
        "history": st.session_state.history,
        "timestamp": datetime.now().isoformat()
    }
    filename = save_session(session_data, f"session_{st.session_state.username}.json")
    st.sidebar.success(f"Saved to {filename}")

uploaded_session = st.sidebar.file_uploader("📂 Load Session", type=["json"])
if uploaded_session:
    session_data = json.load(uploaded_session)
    st.session_state.history = session_data.get("history", [])
    st.sidebar.success("Session loaded!")

# Feature 5: Email report
st.sidebar.markdown("---")
email = st.sidebar.text_input("📧 Email for reports")
if st.sidebar.button("📨 Send Report") and email:
    report_content = f"Classification Report for {st.session_state.username}\n"
    report_content += f"Total predictions: {len(st.session_state.history)}\n"
    for item in st.session_state.history[-5:]:
        report_content += f"- {item['file']}: {item['pred']} ({item['confidence']})\n"
    send_email_report(email, "AI Classification Report", report_content)
    st.sidebar.success("Report sent!")

# Logout button
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# Main area
st.title(txt['title'])
st.markdown(f"*{txt['welcome']}*")

# Statistics Dashboard
if st.checkbox("📊 Show Statistics Dashboard"):
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Images", len(df))
        with col_stat2:
            st.metric("Unique Classes", df['pred'].nunique())
        with col_stat3:
            most_common = df['pred'].mode().iloc[0] if not df.empty else "N/A"
            st.metric("Most Common", most_common)
        with col_stat4:
            avg_conf = df['confidence'].str.replace('%', '').astype(float).mean() if not df.empty else 0
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        
        # Leaderboard
        st.subheader("🏆 Class Leaderboard")
        leaderboard = df['pred'].value_counts().head(10)
        st.bar_chart(leaderboard)
        
        # Confidence distribution
        st.subheader("📈 Confidence Distribution")
        conf_values = df['confidence'].str.replace('%', '').astype(float)
        st.area_chart(conf_values)
    else:
        st.info("No data yet. Upload some images to see statistics!")

# Main upload section
if not camera_enabled and not batch_mode:
    uploaded_files = st.file_uploader(txt['upload'], type=["jpg", "png", "jpeg"], accept_multiple_files=True)
elif batch_mode:
    uploaded_files = batch_files
elif camera_enabled and 'camera_img' in st.session_state:
    uploaded_files = [st.session_state.camera_img]
else:
    uploaded_files = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        start_time = time.time()
        
        col1, col2 = st.columns([1, 1])
        
        # Handle different input types
        if isinstance(uploaded_file, Image.Image):
            img = uploaded_file
            filename = "camera_capture.jpg"
        else:
            img = Image.open(uploaded_file).convert('RGB')
            filename = uploaded_file.name
        
        with col1:
            st.image(img, caption=f"📷 {filename}", use_container_width=True)
        
        probs = get_prediction(model, img)
        top_k_conf, top_k_labels = torch.topk(probs, min(top_k, len(CLASSES)))
        
        with col2:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.subheader(txt['result'])
            
            main_pred = CLASSES[top_k_labels[0]]
            main_conf = top_k_conf[0].item()
            
            st.metric("🎯 Top Prediction", main_pred, f"{main_conf*100:.2f}%")
            st.caption(f"ℹ️ {CLASSES_DESCRIPTION.get(main_pred, 'No description')}")
            
            # Confidence meter
            st.progress(main_conf)
            
            # Top-K Chart
            chart_data = pd.DataFrame({
                'Class': [CLASSES[i] for i in top_k_labels],
                'Confidence': top_k_conf.numpy() * 100
            })
            st.bar_chart(chart_data.set_index('Class'))
            
            # Processing time
            proc_time = time.time() - start_time
            st.caption(f"⏱️ {txt['processing_time']}: {proc_time:.2f} seconds")
            
            # Download button
            result_text = f"Image: {filename}\nPrediction: {main_pred}\nConfidence: {main_conf*100:.2f}%\nDescription: {CLASSES_DESCRIPTION.get(main_pred, '')}"
            st.download_button(txt['download'], result_text, file_name=f"result_{filename}.txt")
            
            # AI Suggestions (Feature 10)
            st.subheader("🤖 AI Suggestions")
            similar_classes = top_k_labels[1:4]
            for idx, class_idx in enumerate(similar_classes, 1):
                st.info(f"💡 Similar to: {CLASSES[class_idx]} - Try uploading an image of this scene!")
            
            if analysis_mode == "Deep (Grad-CAM)" and GRAD_CAM_AVAILABLE:
                with st.spinner('🔥 Generating heatmap...'):
                    heatmap = generate_gradcam(model, img, top_k_labels[0].item())
                    if heatmap is not None:
                        st.image(heatmap, caption="🎨 AI Focus Area (Grad-CAM)", use_container_width=True)
            
            # Feedback
            feedback = st.radio("👍 Was this correct?", ["✅ Yes", "❌ No"], key=f"fb_{filename}", horizontal=True)
            if feedback == "❌ No":
                correct_class = st.selectbox("Correct class?", CLASSES, key=f"correct_{filename}")
                if correct_class:
                    st.session_state.suggestions[filename] = correct_class
                    st.success("Thanks for helping improve the AI!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Save to history
        st.session_state.history.append({
            "file": filename,
            "pred": main_pred,
            "confidence": f"{main_conf*100:.2f}%",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.markdown("---")

# Show history
if st.checkbox("📜 Show History"):
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        # Export statistics
        if st.button("📊 Export Full Statistics"):
            stats = {
                "total_predictions": len(history_df),
                "unique_classes": history_df['pred'].nunique(),
                "most_common": history_df['pred'].mode().iloc[0] if not history_df.empty else None,
                "class_distribution": history_df['pred'].value_counts().to_dict(),
                "user_feedback": st.session_state.suggestions
            }
            st.json(stats)
    else:
        st.info("No history yet. Upload some images!")

# Feature 6: Custom CSS Theme preview
with st.expander("🎨 Custom CSS Theme Editor"):
    custom_css = st.text_area("Write custom CSS:", """
    .stButton>button {
        background-color: #ff4b4b;
        border-radius: 25px;
    }
    """)
    if st.button("Apply Custom CSS"):
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
        st.success("Custom CSS applied!")
