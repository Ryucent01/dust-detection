# Ryucent Dust Detection System - Client Guide

Welcome to your Professional Dust Detection & Labeling System. This package contains everything you need to run the real-time detection on your Jetson device and manage labels via the standalone Studio app.

## Project Structure

- **1_Jetson_Detection/**: Contains the core script and model for the detection hardware.
- **2_Labeling_Studio/**: Contains the standalone `.exe` application for labeling images in the cloud.
- **serviceAccountKey.json**: Your secure key to connect the apps to the Firebase Cloud. **Do not share this file publicly.**

---

## 🚀 1. Real-Time Detection (Jetson Device)
To start the detection on your Jetson:
1. Ensure the Basler GigE camera (or USB camera) is connected and accessible.
2. Open a terminal in the `1_Jetson_Detection` folder.
3. Run: `python dust_detection.py`
4. Use **[SPACE]** to capture and upload images to the cloud for labeling.

## 🎨 2. Ryucent Labeling Studio (PC Application)
To label your data from any Windows PC:
1. Open the `2_Labeling_Studio` folder.
2. Run `RyucentStudio.exe`.
3. Enter your **Access Key** to login.
4. **Draw boxes** on dust particles, then click **[SUBMIT]** (or press 'S') to sync with the cloud.

---

## 🛠️ Support
For technical support or feature requests, contact Ryucent Technologies.
