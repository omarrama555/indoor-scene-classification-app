import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from PIL import Image

# 1. قائمة الفئات بالترتيب الصحيح (67 فئة)
classes = [
    'airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar', 'bathroom', 'bedroom', 'bookstore', 'bowling', 'buffet', 
    'casino', 'children_room', 'church_inside', 'classroom', 'cloister', 'closet', 'clothingstore', 'computerroom', 'concert_hall', 'corridor', 
    'deli', 'dentaloffice', 'dining_room', 'elevator', 'fastfood_restaurant', 'florist', 'gameroom', 'garage', 'greenhouse', 'grocerystore', 
    'gym', 'hairsalon', 'hospitalroom', 'inside_bus', 'inside_subway', 'jewelryshop', 'kitchen', 'laboratorywet', 'laundromat', 'library', 
    'livingroom', 'lobby', 'locker_room', 'mall', 'meeting_room', 'movie_theater', 'museum', 'nursery', 'office', 'operating_room', 
    'pantry', 'poolinside', 'prisoncell', 'restaurant', 'restaurant_kitchen', 'shoeshop', 'stairscase', 'studiomusic', 'subway', 'toystore', 
    'trainstation', 'tv_studio', 'waitingroom', 'warehouse', 'winecellar'
]

# 2. بناء النموذج بنفس هيكل النوبتوك بالضبط
def build_model(num_classes):
    # تحميل الموديل بدون أوزان مسبقة لأننا سنحمل أوزاننا الخاصة
    model = models.efficientnet_b0(weights=None)
    in_f = model.classifier[1].in_features
    # تعديل الـ classifier ليكون Sequential كما حدث في التدريب[cite: 1]
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, num_classes)
    )
    return model

@st.cache_resource
def load_trained_model():
    model = build_model(len(classes))
    # تحميل الأوزان[cite: 1]
    state_dict = torch.load('indoor_model_weights.pth', map_location='cpu')
    
    # تنظيف الأسماء لو كانت تحتوي على 'module.' نتيجة التدريب المتوازي[cite: 1]
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # التحميل مع تعطيل strict لحل مشاكل التسمية البسيطة[cite: 1]
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

# --- واجهة المستخدم ---
st.set_page_config(page_title="Indoor Classifier", page_icon="🏠")
st.title("🏠 Indoor Scene Classifier")
st.write("Upload a photo to identify the indoor environment.")

try:
    model = load_trained_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # التحويلات المطلوبة للصورة[cite: 1]
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
        st.success(f"Prediction: **{classes[pred.item()]}**")
        st.info(f"Confidence: **{conf.item()*100:.2f}%**")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.write("Make sure 'indoor_model_weights.pth' is in your GitHub repository.")
