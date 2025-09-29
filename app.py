

import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 28

class_names = [
    'Apple_Scab_Leaf', 'Apple_leaf', 'Apple_rust_leaf', 'Bell_pepper_leaf',
    'Bell_pepper_leaf_spot', 'Blueberry_leaf', 'Cherry_leaf',
    'Corn_Gray_leaf_spot', 'Corn_leaf_blight', 'Corn_rust_leaf', 'Peach_leaf',
    'Potato_leaf_early_blight', 'Potato_leaf_late_blight', 'Raspberry_leaf',
    'Soyabean_leaf', 'Squash_Powdery_mildew_leaf', 'Strawberry_leaf',
    'Tomato_Early_blight_leaf', 'Tomato_Septoria_leaf_spot', 'Tomato_leaf',
    'Tomato_leaf_bacterial_spot', 'Tomato_leaf_late_blight', 'Tomato_leaf_mosaic_virus',
    'Tomato_leaf_yellow_virus', 'Tomato_mold_leaf', 'Tomato_two_spotted_spider_mites_leaf',
    'grape_leaf', 'grape_leaf_black_rot'
]

diseased_keywords = [
    "blight", "rust", "scab", "spot", "mildew", "virus", "mold", "mites", "black_rot"
]

# Path to local trained model
weights_path = r"D:\demmo\plant\data\best_plantdoc_model.pth"
if not os.path.exists(weights_path):
    st.error(f"❌ Model weights not found at: {weights_path}")


@st.cache_resource(show_spinner=False)
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    model.to(device)
    return model

model = load_model()


transform = transforms.Compose([
    transforms.Resize((160, 160)),  # smaller size for faster CPU inference
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image, or drag and drop it here:")


uploaded_file = st.file_uploader(
    "Choose a leaf image...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=False
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Leaf', use_column_width=True)

 
    input_tensor = transform(image).unsqueeze(0).to(device)

    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        predicted_class = class_names[pred.item()]
        confidence = conf.item() * 100

      
        top3_probs, top3_indices = torch.topk(probs, 3)
        top3_classes = [class_names[i] for i in top3_indices[0]]
        top3_conf = [p.item()*100 for p in top3_probs[0]]

   
    st.subheader("Prediction Result:")
    st.write(f"**Leaf Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    threshold = 50
    if confidence < threshold:
        st.warning(" Model confidence is low. Please try another image or inspect the leaf carefully.")

    if any(keyword.lower() in predicted_class.lower() for keyword in diseased_keywords):
        st.warning(" The leaf appears **unhealthy**.")
        st.info(" Tip: Inspect the plant closely and consider disease-specific treatments such as proper fungicides or pruning affected leaves.")
    else:
        st.success(" The leaf appears **healthy**.")
        st.info(" Tip: Maintain regular irrigation, balanced fertilizers, and proper sunlight exposure.")

    # Top 3 Predictions with progress bars
    st.subheader("Top 3 Predictions:")
    for cls, conf in zip(top3_classes, top3_conf):
        st.progress(int(conf))
        st.write(f"{cls}: {conf:.2f}%")
