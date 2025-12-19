Emotion-Based Music Recommender (Raspberry Pi 4)
This project uses a Convolutional Neural Network (CNN) and MediaPipe to detect real-time facial expressions and recommend music from a local SQLite database based on the detected "vibe". 
It is optimized for the 64-bit Bookworm OS and utilizes TFLite for low-latency inference on embedded hardware.

Key Features
Real-time Inference: Uses a Float32 precision strategy for the TFLite model to maximize the Pi 4's FPU performance.

Stable Face Tracking: Employs MediaPipe for robust detection even in varying lighting conditions.

Indexed Database: Uses SQLite3 for low-latency, randomized song queries mapped to specific emotion categories.

Benchmark Mode: Logs FPS, latency (ms), and confidence percentages to a CSV for performance analysis.

System Requirements
To ensure the math and audio foundations work correctly on the Raspberry Pi, you must install the following system-level dependencies:
sudo apt update
sudo apt install libatlas-base-dev libsdl2-mixer-2.0-0 sqlite3 libv4l-dev -y
Installation & Setup
Clone the Repository:

git clone https://github.com/Tacfulcomb/Emotion-based-Music-Recommender-using-Raspberry-Pi-4.git
cd Emotion-based-Music-Recommender-using-Raspberry-Pi-4

Create a Virtual Environment: Creating a local environment prevents version conflicts with the system-wide Python installation.

python -m venv venv
source venv/bin/activate

Install Pinned Dependencies: This installs the exact versions of MediaPipe, TFLite-runtime, and OpenCV used during development.

pip install -r requirements.txt

How to Run
Ensure your USB webcam and audio output (3.5mm or HDMI) are connected.

python embedded_app_v2.py
Keyboard Controls
SPACE: Start a 3-second manual emotion scan.

V: Toggle VIBE MODE (Continuous scanning and automatic song playback).

S: Stop current music playback.

Q: Safely stop all threads and quit the application.

Project Structure
op_model_float32.tflite: The optimized CNN model file.

music.db: SQLite database containing indexed song metadata.

requirements.txt: List of specific Python library versions for deployment.

