import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import numpy as np

# 1. تعريف قائمة الفئات (يجب أن تكون بنفس ترتيب النوبتوك)
classes = ['airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar', 
           'bathroom', 'bedroom', 'bookstore', 'bowling', 'buffet', 
           'casino', 'children_room', 'church_inside', 'classroom', 'cloister', 
           'closet', 'clothingstore', 'computerroom', 'concert_hall', 'corridor', 
           'deli', 'dentaloffice', 'dining_room', 'elevator', 'fastfood_restaurant', 
           'florist', 'gameroom', 'garage', 'greenhouse', 'grocerystore', 
           'gym', 'hairsalon', 'hospitalroom', 'inside_bus', 'inside_subway', 
           'jewelryshop', 'kitchen', 'laboratorywet', 'laundromat', 'library', 
           'livingroom', 'lobby', 'locker_room', 'mall', 'meeting_room', 
           'movie_theater', 'museum', 'nursery', 'office', 'operating_room', 
           'pantry', 'poolinside', 'prisoncell', 'restaurant', 'restaurant_kitchen', 
           'shoeshop', 'stairscase', 'studiomusic', 'subway', 'toystore', 
           'trainstation', 'tv_studio', 'waitingroom', 'warehouse', 'winecellar']

# 2. دالة بناء النموذج (نسخة طبق الأصل من النوبتوك)
def build_model(num_classes):
    model = models.efficientnet_b0(weights=None) # لا نحتاج لأوزان imagenet هنا لأننا سنحمل أوزاننا
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, num_classes)
    )
    return model

# 3. إعداد الواجهة
st.title("🏠 Indoor Scene Classifier")
st.write("Upload a photo of an indoor place and the AI will tell you what it is!")

# 4. تحميل النموذج والأوزان
@st.cache_resource
def load_trained_model():
    model = build_model(len(classes))
    # تحميل الأوزان مع التعامل مع عدم تطابق الأسماء
    state_dict = torch.load('indoor_model_weights.pth', map_location='cpu')
    
    # إذا كنت قد حفظت النموذج باستخدام DataParallel، الأسماء ستبدأ بكلمة "module."
    # هذا السطر يقوم بحذفها ليتطابق مع النموذج العادي
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # تحميل الأوزان (استخدام strict=False يتجاوز الأخطاء البسيطة في المسميات)
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

model = load_trained_model()

# 5. معالجة الصورة المرفوعة
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # تحويل الصورة لتناسب النموذج
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    # التنبؤ
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        
    st.success(f"Prediction: **{classes[pred.item()]}**")
    st.info(f"Confidence: **{conf.item()*100:.2f}%**")
