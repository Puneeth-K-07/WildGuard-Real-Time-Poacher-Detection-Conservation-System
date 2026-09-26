# 🐾 WildGuard – Real-Time Poacher Detection & Conservation System

WildGuard is an AI-powered wildlife protection system designed to detect potential poachers and suspicious activities in wildlife areas using deep learning and computer vision.

The system uses **YOLOv8** for object detection and **Flask** to provide a web-based interface for uploading images, processing videos, and monitoring detection results.

## 🚀 Features

- Real-time poacher detection
- YOLOv8-based object detection
- Webcam/live camera detection
- Image upload and detection
- Video upload and detection
- Detection result visualization
- Detection logging
- Deforestation/tree-count logging
- Web-based monitoring dashboard
- Sample test images and videos

## 🛠️ Technologies Used

- Python
- YOLOv8
- OpenCV
- Flask
- HTML
- CSS
- CSV
- Computer Vision
- Deep Learning

## 📂 Project Structure

```text
WildGuard-Real-Time-Poacher-Detection-Conservation-System/
│
├── app.py
├── detect_webcam.py
├── requirements.txt
├── README.md
├── LICENSE
├── yolov8n.pt
│
├── models/
│   └── coco.names
│
├── templates/
│   ├── dashboard.html
│   ├── index.html
│   ├── live_camera.html
│   ├── result.html
│   ├── result_view.html
│   ├── show_image.html
│   ├── show_video.html
│   ├── upload_forest.html
│   ├── upload_image.html
│   └── upload_video.html
│
├── test_images/
│
├── deforestation_log.csv
├── detections.csv
└── tree_counts.csv
