import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import google.generativeai as genai
from PIL import Image, ImageDraw
from io import BytesIO
import pyttsx3
from googletrans import Translator
import pytesseract
import streamlit as st
import os

# --- Configuration for Google Generative AI ---
def configure_generative_ai(api_key):
     genai.configure(api_key="AIzaSyDHAIPWQm_GnDpFN9xyLyc3-du22Zx_S6U" )
    
import google.generativeai as genai

def test_generative_api(api_key):
    try:
        genai.configure(api_key="AIzaSyDHAIPWQm_GnDpFN9xyLyc3-du22Zx_S6U")
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(["Test Generative API"])
        print("API Response:", response.text)
    except Exception as e:
        print("Error:", str(e))

# Replace with your API key
test_generative_api("YOUR_API_KEY")


# --- Object Detection Model ---
# Load object detection model
def load_object_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

# COCO class labels (object categories for detection)
COCO_CLASSES = [
    "_background_", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Detect objects in the image
def detect_objects(image, object_detection_model, threshold=0.5):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    predictions = object_detection_model([img_tensor])[0]

    filtered_boxes = [
        (box, label, score)
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores'])
        if score > threshold
    ]
    return filtered_boxes

# Draw bounding boxes on the image
def draw_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    for box, label, score in predictions:
        x1, y1, x2, y2 = box.tolist()
        class_name = COCO_CLASSES[label.item()]
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
        draw.text((x1, y1), f"{class_name} ({score:.2f})", fill="black")
    return image

# --- Generative AI ---
def generate_scene_description(input_prompt, image_data):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content([input_prompt, image_data[0]])
        return response.text
    except Exception as e:
        return f"⚠ Error generating scene description: {str(e)}"

def generate_task_assistance(input_prompt, image_data):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content([input_prompt, image_data[0]])
        return response.text
    except Exception as e:
        return f"⚠ Error generating task assistance: {str(e)}"

# --- Text-to-Speech ---
engine = pyttsx3.init()

def generate_audio_file(text):
    """Generates an audio file from English text and returns it as a BytesIO object."""
    audio = BytesIO()
    try:
        engine.save_to_file(text, "output.mp3")
        engine.runAndWait()
        with open("output.mp3", "rb") as file:
            audio.write(file.read())
        os.remove("output.mp3")  # Cleanup
        audio.seek(0)  # Reset BytesIO pointer
    except Exception as e:
        raise RuntimeError(f"Text-to-Speech Error: {str(e)}")
    return audio

# --- Translation ---
translator = Translator()

def translate_text(text, target_language):
    """Translates text into the specified language."""
    if not target_language:  # If "None" is selected
        return text
    try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        return f"⚠ Translation Error: {str(e)}"

# --- OCR Text Extraction ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust this path for your OS

def extract_text_from_image(image):
    """Extracts text from the given image using OCR."""
    return pytesseract.image_to_string(image)

# --- Streamlit Application ---
def main():
    configure_generative_ai("AIzaSyCfxQdg-kdpYMPCTJu9JoLs7koXeld2Vcs")
    object_detection_model = load_object_detection_model()

    st.set_page_config(page_title="AI Vision Care Assist", page_icon="👁")

    st.markdown("<h3>Select a Feature</h3>", unsafe_allow_html=True)
    feature_choice = st.radio(
        "Choose a feature to interact with:",
        options=["🔍 Describe Scene", "📝 Extract Text", "🚧 Object Detection", "🤖 Personalized Assistance"]
    )

    uploaded_file = st.file_uploader("Upload an Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_data = [{"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}]

        if feature_choice == "🔍 Describe Scene":
            with st.spinner("Generating scene description..."):
                scene_prompt = "Describe the scene."
                scene_response = generate_scene_description(scene_prompt, image_data)
                st.write(scene_response)

        elif feature_choice == "📝 Extract Text":
            with st.spinner("Extracting text using OCR..."):
                extracted_text = extract_text_from_image(image)
                st.write("### Extracted Text:")
                st.text_area("OCR Result", extracted_text, height=200)

        elif feature_choice == "🚧 Object Detection":
            predictions = detect_objects(image, object_detection_model)
            image_with_boxes = draw_boxes(image.copy(), predictions)
            st.image(image_with_boxes, caption="Detected Objects")

        elif feature_choice == "🤖 Personalized Assistance":
            task_prompt = "Assist with tasks based on image data."
            assistance = generate_task_assistance(task_prompt, image_data)
            st.write(assistance)

if __name__ == "__main__":
    main()
