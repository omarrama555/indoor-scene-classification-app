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
from datetime import datetime
from io import BytesIO
import base64

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
        h1, h2, h3, h4, h5, p, span, .stMarkdown {{ color: {theme["text"]}; }}
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
    try:
        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        img_np = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
        input_tensor = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img_pil.resize((224, 224))).unsqueeze(0)
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        return visualization
    except Exception as e:
        return None

# --- 6. دوال الفيتشرات ---
def save_session(session_data, filename="session.json"):
    with open(filename, 'w') as f:
        json.dump(session_data, f)
    return filename

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
if 'camera_img' not in st.session_state:
    st.session_state.camera_img = None

# --- 9. الترجمات ---
TEXTS = {
    "English": {
        "title": "🏢 Pro Indoor AI Explorer Pro",
        "welcome": "Welcome to Indoor Scene Classification",
        "upload": "📤 Upload Image(s)",
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

# Feature 1: Camera Capture (طريقة مبسطة بدون مشاكل)
camera_enabled = st.sidebar.checkbox("📸 Camera Mode")
if camera_enabled:
    st.sidebar.info("📷 Use the camera below to capture an image")
    camera_image = st.camera_input("Take a picture", key="camera_input")
    if camera_image is not None:
        try:
            st.session_state.camera_img = Image.open(camera_image).convert('RGB')
            st.sidebar.success("✅ Image captured successfully!")
        except Exception as e:
            st.sidebar.error(f"Camera error: {e}")

# Feature 2: Batch processing
batch_mode = st.sidebar.checkbox("🔄 Batch Processing Mode")
batch_files = None
if batch_mode:
    batch_files = st.sidebar.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="batch")

# Feature 3: Save/Load session
st.sidebar.markdown("---")
if st.sidebar.button("💾 Save Session"):
    session_data = {
        "username": st.session_state.username,
        "history": st.session_state.history,
        "timestamp": datetime.now().isoformat()
    }
    filename = save_session(session_data, f"session_{st.session_state.username}.json")
    st.sidebar.success(f"✅ Saved to {filename}")

uploaded_session = st.sidebar.file_uploader("📂 Load Session", type=["json"])
if uploaded_session is not None:
    try:
        session_data = json.load(uploaded_session)
        st.session_state.history = session_data.get("history", [])
        st.sidebar.success("✅ Session loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading session: {e}")

# Logout button
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.camera_img = None
    st.rerun()

# Main area
st.title(txt['title'])
st.markdown(f"*{txt['welcome']}*")

# Statistics Dashboard
with st.expander("📊 Statistics Dashboard", expanded=False):
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("📸 Total Images", len(df))
        with col_stat2:
            st.metric("🏷️ Unique Classes", df['pred'].nunique())
        with col_stat3:
            most_common = df['pred'].mode().iloc[0] if not df.empty else "N/A"
            st.metric("🥇 Most Common", most_common)
        with col_stat4:
            try:
                avg_conf = df['confidence'].str.replace('%', '').astype(float).mean() if not df.empty else 0
                st.metric("📈 Avg Confidence", f"{avg_conf:.1f}%")
            except:
                st.metric("📈 Avg Confidence", "N/A")
        
        # Leaderboard
        st.subheader("🏆 Class Leaderboard")
        leaderboard = df['pred'].value_counts().head(10)
        st.bar_chart(leaderboard)
    else:
        st.info("ℹ️ No data yet. Upload some images to see statistics!")

# Main upload section
uploaded_files = []

# تحديد مصدر الصور
if camera_enabled and st.session_state.camera_img is not None:
    uploaded_files = [st.session_state.camera_img]
    st.info("📸 Using camera image")
elif batch_mode and batch_files:
    uploaded_files = batch_files
else:
    uploaded_files = st.file_uploader(txt['upload'], type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="main_uploader")

# معالجة الصور
if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        start_time = time.time()
        
        col1, col2 = st.columns([1, 1])
        
        # Handle different input types
        if isinstance(uploaded_file, Image.Image):
            img = uploaded_file
            filename = f"camera_capture_{idx}.jpg"
        else:
            try:
                img = Image.open(uploaded_file).convert('RGB')
                filename = uploaded_file.name
            except Exception as e:
                st.error(f"Error opening image: {e}")
                continue
        
        with col1:
            st.image(img, caption=f"📷 {filename}", use_container_width=True)
        
        # Get prediction
        try:
            probs = get_prediction(model, img)
            top_k_conf, top_k_labels = torch.topk(probs, min(top_k, len(CLASSES)))
            
            with col2:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.subheader(txt['result'])
                
                main_pred = CLASSES[top_k_labels[0]]
                main_conf = top_k_conf[0].item()
                
                st.metric("🎯 Top Prediction", main_pred, f"{main_conf*100:.2f}%")
                st.caption(f"ℹ️ {CLASSES_DESCRIPTION.get(main_pred, 'No description available')}")
                
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
                result_text = f"Image: {filename}\nPrediction: {main_pred}\nConfidence: {main_conf*100:.2f}%\nDescription: {CLASSES_DESCRIPTION.get(main_pred, '')}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                st.download_button(txt['download'], result_text, file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                
                # AI Suggestions
                st.subheader("🤖 AI Suggestions")
                similar_classes = top_k_labels[1:4]
                for class_idx in similar_classes:
                    st.info(f"💡 Try: {CLASSES[class_idx]}")
                
                # Grad-CAM
                if analysis_mode == "Deep (Grad-CAM)" and GRAD_CAM_AVAILABLE:
                    with st.spinner('🔥 Generating heatmap...'):
                        heatmap = generate_gradcam(model, img, top_k_labels[0].item())
                        if heatmap is not None:
                            st.image(heatmap, caption="🎨 AI Focus Area (Grad-CAM)", use_container_width=True)
                        else:
                            st.warning("⚠️ Grad-CAM unavailable for this image")
                
                # Feedback
                feedback = st.radio("👍 Was this correct?", ["✅ Yes", "❌ No"], key=f"fb_{filename}_{idx}", horizontal=True)
                if feedback == "❌ No":
                    correct_class = st.selectbox("What is the correct class?", CLASSES, key=f"correct_{filename}_{idx}")
                    if correct_class:
                        st.session_state.suggestions[filename] = correct_class
                        st.success("🙏 Thanks for helping improve the AI!")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Save to history
            st.session_state.history.append({
                "file": filename,
                "pred": main_pred,
                "confidence": f"{main_conf*100:.2f}%",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.markdown("---")
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            continue

# Show history
with st.expander("📜 Classification History", expanded=False):
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        # Export all history
        if st.button("📥 Export Full History as CSV"):
            csv = history_df.to_csv(index=False)
            st.download_button("Download CSV", csv, file_name=f"ai_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    else:
        st.info("ℹ️ No history yet. Upload some images!")

# Custom CSS Theme Editor
with st.expander("🎨 Custom CSS Theme Editor"):
    custom_css = st.text_area("Write custom CSS:", height=150, placeholder=".stButton>button {\n    background-color: #ff4b4b;\n    border-radius: 25px;\n}")
    if st.button("🎨 Apply Custom CSS"):
        if custom_css.strip():
            st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
            st.success("✅ Custom CSS applied successfully!")
        else:
            st.warning("⚠️ Please enter some CSS first!")
