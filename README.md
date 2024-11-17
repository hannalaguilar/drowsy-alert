# **Drowsy-Alert** <img src="images/logo2.png" alt="Logo" width="15%">

A lightweight drowsiness detection system using Mediapipe's **Face Landmark Detection Model** and a **RandomForestClassifier** to classify states like `normal` or `yawning` based on facial blendshapes. It supports real-time webcam inference and video processing.

<img src="images/example.png" alt="Logo" width="50%">

## **Quick Setup**

Create an enviroment:
   ```bash
   git clone https://github.com/hannalaguilar/drowsy-alert
   cd drowsy-alert
   conda create --name drowsy-alert python=3.10
   conda activate drowsy-alert
   pip install -r requirements.txt
   ````

## Run Inference

The trained model is stored in the following path: `models/random_forest_model_1.pkl`

### 1. Real-Time via Webcam

```bash
python inference_webcam.py
```

### 2. Pre-Recorded Video

```bash
python inference_video.py --video_path path/to/video.mp4
```
