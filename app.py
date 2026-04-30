import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from PIL import Image

# 1. القائمة المستخرجة من النوتبوك (67 فئة بالترتيب الصحيح)
classes = ['airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar', 'bathroom', 'bedroom', 'bookstore', 'bowling', 'buffet', 'casino', 'children_room', 'church_inside', 'classroom', 'cloister', 'closet', 'clothingstore', 'computerroom', 'concert_hall', 'corridor', 'deli', 'dentaloffice', 'dining_room', 'elevator', 'fastfood_restaurant', 'florist', 'gameroom', 'garage', 'greenhouse', 'grocerystore', 'gym', 'hairsalon', 'hospitalroom', 'inside_bus', 'inside_subway', 'jewelleryshop', 'kindergarden', 'kitchen', 'laboratorywet', 'laundromat', 'library', 'livingroom', 'lobby', 'locker_room', 'mall', 'meeting_room', 'movietheater', 'museum', 'nursery', 'office', 'operating_room', 'pantry', 'poolinside', 'prisoncell', 'restaurant', 'restaurant_kitchen', 'shoeshop', 'stairscase', 'studiomusic', 'subway', 'toystore', 'trainstation', 'tv_studio', 'videostore', 'waitingroom', 'warehouse', 'winecellar']

# 2. بناء هيكل النموذج (مطابق تماماً لخلية بناء الموديل في النوتبوك)
def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, num_classes)
    )
    return model

@st.cache_resource
def load_trained_model():
    model = build_model(len(classes))
    # تحميل الأوزان مع التعامل مع الـ CPU
    state_dict = torch.load('indoor_model_weights.pth', map_location='cpu')
    
    # تنظيف الأسماء لو كان التدريب تم بـ DataParallel[cite: 1]
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

# --- واجهة المستخدم (GUI) ---
st.set_page_config(page_title="Indoor Classifier", page_icon="🏠")
st.title("🏠 Indoor Scene Classifier")
st.write("Upload a photo to identify the indoor environment.")

try:
    model = load_trained_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # المعالجة المسبقة للصورة (Resize & Normalize)[cite: 1]
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        
        # مرحلة التنبؤ[cite: 1]
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
        st.success(f"Prediction: **{classes[pred.item()]}**")
        st.info(f"Confidence: **{conf.item()*100:.2f}%**")

except Exception as e:
    st.error(f"Something went wrong: {e}")
