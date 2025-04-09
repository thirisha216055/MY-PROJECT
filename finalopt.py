import streamlit as st
import numpy as np
import plotly.graph_objects as go
import zipfile
import os
import glob
import shutil
import random
import torch
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model (for furniture detection)
model = YOLO("yolov8n.pt")

# Predefined room types
ROOM_TYPES = ["Bathroom", "Bedroom", "Dining", "Kitchen", "Living Room", "Office", "Other"]

# Set page configuration
st.set_page_config(page_title="AI Interior Design Assistant", layout="wide")

st.title("🏡 AI Interior Design Assistant")

# Sidebar: Room dimensions
st.sidebar.header("Room Dimensions")
length = st.sidebar.number_input("Length (m)", min_value=2.0, max_value=10.0, value=5.0, step=0.5)
width = st.sidebar.number_input("Width (m)", min_value=2.0, max_value=10.0, value=4.0, step=0.5)
height = st.sidebar.number_input("Height (m)", min_value=2.0, max_value=5.0, value=2.5, step=0.5)

# Upload dataset
st.sidebar.header("Upload Dataset")
uploaded_zip = st.sidebar.file_uploader("Choose a ZIP file", type="zip", key="zip_uploader")

# Select room type
st.sidebar.header("Select Room Type")
room_type = st.sidebar.selectbox("Choose a room:", ROOM_TYPES)

# Function to extract ZIP files
def extract_zip(uploaded_file, extract_to='datasets/'):
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(uploaded_file, 'r') as z:
        z.extractall(extract_to)

    return extract_to

# Function to find images in a selected room type
def find_images(directory, room_type):
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    
    return [img for img in image_files if room_type.lower() in img.lower()]

# Function to detect furniture using YOLOv8
def detect_furniture(image_path):
    results = model(image_path)
    detected_furniture = []
    image = Image.open(image_path)
    img_width, img_height = image.size

    for result in results:
        for box in result.boxes:
            cls = result.names[int(box.cls)]  # Get class name (e.g., 'chair', 'table')
            x_center, y_center = box.xywh[0][:2]  # Get center of bounding box
            x_3d = (x_center / img_width) * length  # Scale to room length
            y_3d = (y_center / img_height) * width  # Scale to room width
            detected_furniture.append((cls, x_3d, y_3d, 0.5, 'red'))  # Default height & color

    return detected_furniture

# Function to create a 3D room with detected furniture
def create_3d_room_with_furniture(length, width, height, detected_furniture):
    fig = go.Figure()

    # Define room walls
    x = np.linspace(0, length, 10)
    y = np.linspace(0, width, 10)
    X, Y = np.meshgrid(x, y)
    Z_floor = np.zeros_like(X)

    # Add floor and walls
    fig.add_trace(go.Surface(x=X, y=Y, z=Z_floor, colorscale='gray', showscale=False))  # Floor
    fig.add_trace(go.Surface(x=X, y=np.zeros_like(X), z=X*0+height, colorscale='gray', opacity=0.5))  # Left wall
    fig.add_trace(go.Surface(x=X, y=Y[-1], z=X*0+height, colorscale='gray', opacity=0.5))  # Right wall
    fig.add_trace(go.Surface(x=np.zeros_like(Y), y=Y, z=X*0+height, colorscale='gray', opacity=0.5))  # Back wall

    # Add detected furniture
    for item in detected_furniture:
        name, x_pos, y_pos, z_pos, color = item
        fig.add_trace(go.Scatter3d(x=[x_pos], y=[y_pos], z=[z_pos], mode='markers',
                                   marker=dict(size=5, color=color), name=name))

    # Layout settings
    fig.update_layout(
        scene=dict(
            xaxis_title='Length',
            yaxis_title='Width',
            zaxis_title='Height',
            aspectmode='manual',
            aspectratio=dict(x=length/5, y=width/5, z=height/5)
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

# Process uploaded ZIP and display images
if st.button("Load Dataset") or st.session_state.get("dataset_loaded"):
    if uploaded_zip is not None:
        extracted_path = extract_zip(uploaded_zip)
        st.session_state["dataset_loaded"] = True
        images = find_images(extracted_path, room_type)

        if images:
            st.subheader(f"Random {room_type} Image")
            selected_image_path = random.choice(images)
            selected_image = Image.open(selected_image_path)
            st.image(selected_image, use_column_width=True)

            # Detect furniture in the selected image
            detected_furniture = detect_furniture(selected_image_path)
        else:
            st.write(f"No images found for the selected room type ({room_type}).")
    else:
        st.write("Please upload a dataset.")

# Generate and display the 3D room layout
st.subheader("3D Room Layout with Detected Furniture")
fig = create_3d_room_with_furniture(length, width, height, detected_furniture if 'detected_furniture' in locals() else [])
st.plotly_chart(fig)

# Refresh button
if st.button("Refresh Page"):
    st.session_state["dataset_loaded"] = False
    st.rerun()
