 🏠 AI Interior Design Recommendation System

The AI Interior Design Recommendation System is a smart, ML-powered application designed to assist users in selecting suitable furniture and décor items based on room dimensions, preferred style, and budget. It integrates computer vision, recommendation algorithms, and 3D visualization to personalize and simplify the interior design process for users.

 🔍 Project Objectives

- Recommend furniture items based on user-provided room dimensions, style preferences, and budget.
- Classify interior styles and types from input images using a trained deep learning model (YOLOv8).
- Visualize furniture placement in rooms using interactive 3D/AR views.
- Deliver a smooth user experience via an intuitive Streamlit-based web interface.

 🧠 Technologies Used

- Python – Core logic, data processing, and ML integration
- YOLOv8 – For object detection and style/type classification in interior images
- Pandas & NumPy – Data preprocessing and manipulation
- SQLite – Local database to store furniture metadata
- Plotly – 3D visualization of furniture in a room
- Streamlit – Web app frontend and user input interface

 📦 Datasets Used

- SUN RGB-D 2D Dataset – For training the model on interior scenes and layout recognition
- Furniture Metadata Dataset – Includes furniture types, dimensions, categories, and compatible style tags

 ⚙️ Features

- ✅ Upload a room image to detect interior style and classify layout type
- ✅ Input room width, length, and budget to receive personalized furniture recommendations
- ✅ Filter furniture by design style (e.g., Modern, Minimalist, Industrial, etc.)
- ✅ Visualize furniture recommendations in a 3D plot using Plotly
- ✅ Lightweight and interactive interface built using Streamlit

 🚀 How It Works

1. User uploads a room image or selects a style manually.
2. YOLOv8 model detects the room style and layout from the image.
3. Based on input dimensions and budget, a filtered list of furniture is generated.
4. Recommended items are displayed with dimensions, images, and placement suggestions.
5. Users can explore furniture placement through interactive 3D plots.
   
![WhatsApp Image 2025-04-23 at 10 36 53_9ca3ba90](https://github.com/user-attachments/assets/3cfcd0ad-b2d0-4a45-850b-777d8000a4bd)


![WhatsApp Image 2025-04-23 at 10 42 11_194dd7a1](https://github.com/user-attachments/assets/3abddbbe-ff31-481d-9bfa-bdbb5af7bace)

![WhatsApp Image 2025-04-23 at 10 42 36_dab1ba0d](https://github.com/user-attachments/assets/458e774c-80e6-4983-94a3-c4dbb8435eb5)


![WhatsApp Image 2025-04-23 at 10 35 55_c0a16b27](https://github.com/user-attachments/assets/65d0a1b8-459a-400e-a8ba-d392fa70ebb2)
