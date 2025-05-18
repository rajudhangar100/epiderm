# AI-Powered Skin Disease Detection System (Offline Edge AI)

An AI-based, fully offline *Skin Disease and Early Cancer Detection System, designed for **real-time diagnosis* using a *Convolutional Neural Network (CNN)* model deployed on a *Raspberry Pi*. The system captures dermatoscopic images, analyzes them using an embedded TensorFlow model, and displays both the prediction and a Grad-CAM heatmap to highlight the region of concern — providing an intuitive and interpretable early screening solution at the edge.

## 💡 Overview

Skin cancer, especially melanoma, can be life-threatening if not detected early. Our system aims to *empower early detection* through an *AI-driven embedded device, enabling proactive diagnosis **without internet dependency. Designed to run entirely **offline, this application leverages **edge computing* to bring medical-grade assistance to remote and underserved areas — right from a *Raspberry Pi with touchscreen and dermatoscope camera*.

## 🎯 Key Features

- ✅ *Early skin cancer detection* using a trained CNN model.
- 📸 *Real-time dermatoscopic image capture* with a connected camera.
- 🧠 *Edge AI deployment* using TensorFlow 2.18.0 — runs fully offline.
- 🔍 *Grad-CAM heatmap visualization* to interpret model predictions.
- 🖥 *Touchscreen-optimized fullscreen GUI* (480x320 resolution).
- 🗂 Modular and intuitive interface with Home, Capture, About, and Discard functions.

## 🧠 AI Model Details

- *Model Type*: Convolutional Neural Network (CNN)
- *Training Dataset*: Skin lesion images including benign and malignant types (e.g., ISIC archive)
- *Deployment Format*: .h5 file loaded with TensorFlow 2.18.0
- *Explanation Tool*: Grad-CAM (Gradient-weighted Class Activation Mapping)

  ## 🔧 Installation

### 📌 Prerequisites

- Raspberry Pi 4 or higher
- 480x320 LCD touchscreen (preconfigured)
- Connected dermatoscope or USB camera

### 🐍 Python & Library Installation

Install Python 3.8+ and required libraries:

```bash
sudo apt update
sudo apt install python3-pip python3-tk libatlas-base-dev

Install Python dependencies:
pip install tensorflow==2.18.0
pip install opencv-python
pip install matplotlib
pip install numpy
pip install Pillow

Running the Application
Clone this repository:

git clone https://github.com/rajudhangar100/epiderm

Place your trained .h5 model inside the models/ folder.

Run the main script:

python3 skin_gui.py
The application will launch in fullscreen mode. Use the GUI to:

Capture an image of the skin.

Classify using the embedded CNN model.

Visualize results with Grad-CAM.

Navigate using Home, About, and Discard options.

Privacy & Security
This system runs completely offline — no patient data is uploaded or stored externally.

All captured images are stored temporarily and can be discarded easily from the GUI.

Ideal for deployment in remote clinics and rural healthcare setups where data privacy and offline operation are crucial.
