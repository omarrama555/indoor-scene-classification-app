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
from datetime import datetime

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False

# --- 1. إعدادات الصفحة والـ CSS المخصص ---
st.set_page_config(page_title="Pro Indoor AI Explorer", page_icon="🏢", layout="wide")

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

# --- CSS مخصص مع وضع Dark/Light mode ---
def apply_theme(theme):
    if theme == "Dark":
        st.markdown("""
            <style>
            .main { background-color: #1e1e2f; color: white; }
            .stButton>button { background-color: #ff4b4b; color: white; border-radius: 20px; }
            .prediction-card { background-color: #2d2d44; border-radius: 15px; padding: 15px; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .main { background-color: #f0f2f6; }
            .stButton>button { background-color: #ff4b4b; border-radius: 20px; }
            .prediction-card { background-color: white; border-radius: 15px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            </style>
        """, unsafe_allow_html=True)

# --- 3. بناء وتحميل النموذج مع Progress Bar ---
def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    return model

@st.cache_resource
def load_model():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 جاري تحميل النموذج... 0%")
    model = build_model(len(CLASSES))
    progress_bar.progress(20)
    
    status_text.text("📥 جاري تحميل الأوزان... 40%")
    try:
        state_dict = torch.load('indoor_model_weights.pth', map_location='cpu')
        progress_bar.progress(70)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        status_text.text("✅ تم تحميل النموذج بنجاح!")
    except FileNotFoundError:
        st.warning("⚠️ ملف النموذج غير موجود! سيتم استخدام نموذج فارغ (للاختبار فقط)")
        status_text.text("⚠️ نموذج فارغ - حمّل الأوزان أولاً")
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {e}")
        status_text.text("❌ فشل تحميل النموذج")
    
    progress_bar.progress(100)
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
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
    if not GRAD_CAM_AVAILABLE:
        return None
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    img_np = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    input_tensor = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img_pil.resize((224, 224))).unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization

# --- 5. دوال مساعدة جديدة ---
def confidence_meter(confidence):
    return st.progress(float(confidence))

def search_classes(search_term):
    if search_term:
        return [c for c in CLASSES if search_term.lower() in c.lower()]
    return CLASSES

# ==================== تحميل النموذج أولاً ====================
# ده المكان اللي بيتحمل فيه النموذج قبل أي حاجة تانية
try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ فشل تحميل النموذج: {e}")
    st.info("تأكد من وجود ملف 'indoor_model_weights.pth' في نفس المجلد")
    st.stop()

# ==================== باقي الواجهة ====================

# نظام login بسيط
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.title("🔐 Welcome to Pro Indoor AI")
    username_input = st.text_input("👤 Enter your name to start:")
    if username_input:
        st.session_state.username = username_input
        st.session_state.logged_in = True
        st.rerun()
    st.stop()

# Sidebar - Control Panel
st.sidebar.title(f"👋 Hello, {st.session_state.username}!")
st.sidebar.markdown("---")

# Theme selector
theme = st.sidebar.selectbox("🎨 Theme", ["Light", "Dark"])
apply_theme(theme)

# Analysis settings
analysis_mode = st.sidebar.selectbox("🔬 Analysis Depth", ["Standard", "Deep (Grad-CAM)"])
top_k = st.sidebar.selectbox("📊 Number of Top Predictions", [3, 5, 10], index=1)
show_history = st.sidebar.checkbox("📜 Show History", value=True)

# Search filter
st.sidebar.markdown("---")
search_term = st.sidebar.text_input("🔍 Filter Classes")
filtered_classes = search_classes(search_term)
st.sidebar.caption(f"📋 {len(filtered_classes)} classes loaded")

# Buttons
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []
    st.rerun()

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# Main content
st.title(f"🏢 Pro Indoor AI Classifier")
st.markdown(f"*Welcome {st.session_state.username}! Upload an image to classify indoor scenes*")

# --- مقارنة صورتين ---
compare_mode = st.checkbox("🔄 Compare two images")

if compare_mode:
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        st.subheader("Image 1")
        file1 = st.file_uploader("Upload first image", type=["jpg", "png", "jpeg"], key="img1")
    with col_comp2:
        st.subheader("Image 2")
        file2 = st.file_uploader("Upload second image", type=["jpg", "png", "jpeg"], key="img2")
    
    if file1 and file2:
        img1 = Image.open(file1).convert('RGB')
        img2 = Image.open(file2).convert('RGB')
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(img1, caption="Image 1", use_container_width=True)
        with col_img2:
            st.image(img2, caption="Image 2", use_container_width=True)
        
        with st.spinner("Analyzing both images..."):
            probs1 = get_prediction(model, img1)
            probs2 = get_prediction(model, img2)
            pred1 = CLASSES[torch.argmax(probs1)]
            pred2 = CLASSES[torch.argmax(probs2)]
            
            st.info(f"📌 Image 1 Prediction: **{pred1}**")
            st.info(f"📌 Image 2 Prediction: **{pred2}**")
else:
    uploaded_files = st.file_uploader("📤 Upload Image(s)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if 'history' not in st.session_state: 
            st.session_state.history = []
        
        for uploaded_file in uploaded_files:
            start_time = time.time()
            
            col1, col2 = st.columns([1, 1])
            img = Image.open(uploaded_file).convert('RGB')
            
            with col1:
                st.image(img, caption=f"📷 {uploaded_file.name}", use_container_width=True)
            
            probs = get_prediction(model, img)
            top_k_conf, top_k_labels = torch.topk(probs, min(top_k, len(CLASSES)))
            
            with col2:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.subheader("📊 Analysis Results")
                
                main_pred = CLASSES[top_k_labels[0]]
                main_conf = top_k_conf[0].item()
                
                st.metric("🎯 Top Prediction", main_pred, f"{main_conf*100:.2f}%")
                st.caption(f"ℹ️ {CLASSES_DESCRIPTION.get(main_pred, 'وصف غير متوفر')}")
                
                # Confidence meter
                st.write("📈 Confidence Meter:")
                confidence_meter(main_conf)
                
                # Top-K Chart
                chart_data = pd.DataFrame({
                    'Class': [CLASSES[i] for i in top_k_labels],
                    'Confidence': top_k_conf.numpy() * 100
                })
                st.bar_chart(chart_data.set_index('Class'))
                
                # Processing time
                proc_time = time.time() - start_time
                st.caption(f"⏱️ Processing time: {proc_time:.2f} seconds")
                
                # Download button
                st.download_button(
                    label="📥 Download Result",
                    data=f"Image: {uploaded_file.name}\nPrediction: {main_pred}\nConfidence: {main_conf*100:.2f}%\nDescription: {CLASSES_DESCRIPTION.get(main_pred, '')}",
                    file_name=f"result_{uploaded_file.name}.txt"
                )
                
                if analysis_mode == "Deep (Grad-CAM)" and GRAD_CAM_AVAILABLE:
                    with st.spinner('🔥 Generating heatmap...'):
                        heatmap = generate_gradcam(model, img, top_k_labels[0].item())
                        if heatmap is not None:
                            st.image(heatmap, caption="🎨 AI Focus Area (Grad-CAM)", use_container_width=True)
                        else:
                            st.warning("Grad-CAM not available")
                
                # User feedback
                feedback = st.radio("👍 Was this prediction correct?", ["✅ Yes", "❌ No", "🤔 Not sure"], key=f"fb_{uploaded_file.name}", horizontal=True)
                if feedback == "✅ Yes":
                    st.success("Thanks for confirming!")
                elif feedback == "❌ No":
                    correct_class = st.selectbox("What is the correct class?", CLASSES, key=f"correct_{uploaded_file.name}")
                    if correct_class:
                        st.info(f"Noted! Correct class: {correct_class}")
                
                st.markdown('</div>', unsafe_allow_html=True)

            # Add to history
            st.session_state.history.append({
                "file": uploaded_file.name, 
                "pred": main_pred,
                "confidence": f"{main_conf*100:.2f}%",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.markdown("---")

# Show History
if show_history and 'history' in st.session_state and st.session_state.history:
    st.markdown("## 📜 Classification History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
    
    # Export options
    if st.button("📩 Export as CSV"):
        csv = history_df.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name=f"ai_history_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    
    if st.button("🗑️ Clear All History"):
        st.session_state.history = []
        st.rerun()
