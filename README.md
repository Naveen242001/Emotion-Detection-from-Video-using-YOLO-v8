**Emotion Detection from Video Using YOLOv8
This repository implements a robust real-time emotion detection system using YOLOv8 for face detection and emotion classification. The project combines object detection and deep learning-based emotion recognition to process recorded or live video streams.**

**Features**
YOLOv8 for Face Detection:
Fine-tuned on the WIDER FACE dataset for accurate face detection.
**Emotion Classification:**
Trained on datasets like FER-2013 and AffectNet for multi-class emotion detection.
Emotion categories include: happy, sad, angry, surprised, neutral, and more.
Real-Time Processing:
Handles live video streams or recorded files.
Displays bounding boxes around detected faces along with their predicted emotions.
**Deployment-Ready:**
Easily deployable on GPU-enabled cloud platforms.
Includes a REST API for integrating with external applications.
Interactive Web Interface:
A simple web application to visualize live results in real-time.
**Pipeline Overview**
Face Detection:
Detect faces in video frames using YOLOv8.
Emotion Classification:
Predict emotions using a custom-trained emotion classification model.
Integration:
Combine face detection and emotion classification into a seamless pipeline.
**Usage**
Train the YOLOv8 model for face detection.
Train the emotion classification model on labeled datasets.
Deploy the system using the provided REST API or interactive web app.
**Technologies Used**
YOLOv8: For efficient face detection.
PyTorch: Framework for training emotion classification models.
OpenCV: For video processing.
Flask/FastAPI: For REST API development.
HTML/CSS/JavaScript: For the web interface.
**Applications**
Real-time emotion recognition in human-computer interaction.
Video analysis for emotion-based insights.
Behavioral studies and sentiment analysis.
