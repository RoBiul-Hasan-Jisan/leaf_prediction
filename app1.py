import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import os
import cv2
import tempfile
from collections import Counter

# -------------------------------
# Streamlit Page Config (Must be First)
# -------------------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")

# -------------------------------
# Device and Model Setup
# -------------------------------
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

weights_path = "data/best_plantdoc_model.pth"
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
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("🌿 Plant Disease Detector")
st.write("Upload a **leaf image** or a **leaf video** for disease detection.")

# -------------------------------
# Select Input Type
# -------------------------------
option = st.radio("Select input type:", ["Image", "Video"])

# -------------------------------
# Image Processing
# -------------------------------
if option == "Image":
    uploaded_file = st.file_uploader(
        "Choose a leaf image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Leaf', use_container_width=True)

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            predicted_class = class_names[pred.item()]
            confidence = conf.item() * 100

            top3_probs, top3_indices = torch.topk(probs, 3)
            top3_classes = [class_names[i] for i in top3_indices[0]]
            top3_conf = [p.item() * 100 for p in top3_probs[0]]

        st.subheader("🖼️ Image Prediction Result:")
        st.write(f"**Leaf Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        if any(keyword in predicted_class.lower() for keyword in diseased_keywords):
            st.warning("🌱⚠️ The leaf appears **unhealthy**")
        else:
            st.success("✅ The leaf appears **healthy**")

        st.subheader("Top 3 Predictions (Image):")
        for cls, conf in zip(top3_classes, top3_conf):
            st.progress(int(conf))
            st.write(f"{cls}: {conf:.2f}%")

# -------------------------------
# Video Processing
# -------------------------------
if option == "Video":
    uploaded_video = st.file_uploader(
        "Choose a leaf video...",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        st.subheader("🎥 Video Prediction Result:")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_count = 0
        frame_predictions = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % 10 == 0:  # process every 10th frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    conf, pred = torch.max(probs, 1)
                    predicted_class = class_names[pred.item()]
                    confidence = conf.item() * 100
                    frame_predictions.append(predicted_class)

                cv2.putText(frame, f"{predicted_class} ({confidence:.1f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()
        cv2.destroyAllWindows()

        # Cleanup temp file safely
        try:
            os.remove(tfile.name)
        except PermissionError:
            pass

        # -------------------------------
        # Video Summary Result
        # -------------------------------
        if frame_predictions:
            most_common = Counter(frame_predictions).most_common(1)[0]
            final_class, count = most_common
            st.subheader("📊 Video Summary Result:")
            st.write(f"**Most Detected Class:** {final_class}")
            st.write(f"**Appeared in {count} frames out of {len(frame_predictions)} processed frames**")

            if any(keyword in final_class.lower() for keyword in diseased_keywords):
                st.warning("🌱⚠️ Overall, the leaf appears **unhealthy**")
            else:
                st.success("✅ Overall, the leaf appears **healthy**")
