# Ryucent Dust Detection System - Client Guide

Welcome to your Professional Dust Detection & Labeling System. This repository contains the source code for real-time detection on Jetson devices and the labeling studio application.

## Project Structure

- **detection/**: Contains the core script (`dust_detection.py`) and dependencies for the detection hardware.
- **studio/**: Contains the source code for the Ryucent Labeling Studio (`ryucent_studio.py`).
- **models/**: Contains the AI model file (`best_dust.pt`).
- **README.md**: This guide.

## 🔑 Security Requirement

> [!IMPORTANT]
> **serviceAccountKey.json is EXCLUDED**: For security reasons, the Firebase credentials file is not included in this repository. 
> 1. Obtain your `serviceAccountKey.json` from the Firebase Console.
> 2. Place it in the **root** of this project folder so both the detection and studio scripts can access it.

---

## 🚀 1. Real-Time Detection (Jetson Device)
To start the detection on your Jetson:
1. Ensure the USB camera is connected.
2. Install dependencies: `pip install -r detection/requirements.txt`
3. Run: `python detection/dust_detection.py`
4. Use **[SPACE]** to capture and upload images to the cloud for labeling.

## 🎨 2. Ryucent Labeling Studio
To run the labeling studio from source:
1. Ensure you have the dependencies installed.
2. Run: `python studio/ryucent_studio.py`
3. Enter your **Access Key** to login.
4. **Draw boxes** on dust particles, then click **[SUBMIT]** (or press 'S') to sync with the cloud.

---

## 🛠️ Support
For technical support or feature requests, contact Ryucent Technologies.