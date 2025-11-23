import numpy as np
import cv2
import sqlite3
import os
import time
import psutil
from collections import Counter
from datetime import datetime
import glob

# Use tflite_runtime for inference
import tflite_runtime.interpreter as tflite

# Use Pygame for Audio
import pygame

# --- Configuration ---
TFLITE_MODEL_PATH = 'op_model_float32.tflite'
SQLITE_DB_PATH = 'music.db'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
DB_TABLE_NAME = 'songs'
TEST_IMAGE_FOLDER = 'test_images'

# --- Emotion Mapping ---
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# --- Database Function ---
def get_recommendations_from_db(emotion_list):
    recommendations = []
    conn = None
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        print(f"\n--- Querying recommendations for emotions: {emotion_list} ---")

        num_emotions = len(emotion_list)
        if num_emotions == 1:
            times = [10]
        elif num_emotions == 2:
            times = [7, 3]
        else:
            times = [5, 3, 2]

        for i, emotion in enumerate(emotion_list):
            limit = times[i] if i < len(times) else times[-1]
            if limit <= 0: continue

            # We select 'link' which is the file path
            query = f"""
                SELECT name, artist, link 
                FROM {DB_TABLE_NAME}
                WHERE emotion_category = ?
                AND link IS NOT NULL AND link != ''
                ORDER BY RANDOM()
                LIMIT ?
            """
            cursor.execute(query, (emotion, limit))
            songs = cursor.fetchall()
            print(f"   -> Found {len(songs)} songs for '{emotion}'")
            recommendations.extend(songs)

        conn.close()
        np.random.shuffle(recommendations)
        return recommendations[:10] # Return max 10 songs

    except Exception as e:
        print(f"❌ Database error: {e}")
        if conn: conn.close()
        return []

# --- Emotion Processing ---
def pre(l):
    if not l: return []
    try:
        emotion_counts = Counter(l)
        sorted_emotions = sorted(emotion_counts.items(), key=lambda item: (-item[1], item[0]))
        return [emotion for emotion, count in sorted_emotions]
    except Exception as e:
        return []

# --- Main Test Function ---
def main_test():
    print("--- Test Script Initializing ---")

    # 1. Initialize Audio
    try:
        pygame.mixer.init()
        print("✅ Audio Mixer Initialized")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize audio: {e}")

    # 2. Load Models
    try:
        interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        input_type = input_details[0]['dtype']
        
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade.empty(): raise IOError("Cannot load Haarcascade")
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # 3. Find Images
    image_files = glob.glob(os.path.join(TEST_IMAGE_FOLDER, '*.*'))
    valid_exts = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in valid_exts]

    if not image_files:
        print(f"❌ No images found in {TEST_IMAGE_FOLDER}")
        return

    detected_emotions_list = []
    
    # 4. Process Images
    print(f"\n--- Processing {len(image_files)} images ---")
    for image_path in image_files:
        print(f"Processing: {os.path.basename(image_path)}")
        frame = cv2.imread(image_path)
        if frame is None: continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi_gray = gray[y:y + h, x:x + w]
            
            img_resized = cv2.resize(roi_gray, (width, height))
            img_expanded = np.expand_dims(img_resized, axis=-1)
            input_data = img_expanded.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            max_index = int(np.argmax(output_data[0]))
            emotion = emotion_dict.get(max_index, "Unknown")
            conf = output_data[0][max_index]
            
            detected_emotions_list.append(emotion)
            print(f"   -> Detected: {emotion} ({conf:.2f})")
        else:
            print("   -> No face detected")

    # 5. Get Recommendations
    unique_emotions = pre(detected_emotions_list)
    recommendations = []
    if unique_emotions:
        recommendations = get_recommendations_from_db(unique_emotions)

    # 6. Interactive Player Loop
    print("\n" + "="*40)
    print("      🎵  RECOMMENDATIONS FOUND  🎵")
    print("="*40)
    
    if not recommendations:
        print("No recommendations found.")
        return

    # Print the list once
    for i, (name, artist, link) in enumerate(recommendations):
        print(f" [{i+1}] {name} - {artist}")

    print("\nCommands:")
    print(" - Type a NUMBER (1-10) to play a song")
    print(" - Type 's' to STOP playing")
    print(" - Type 'q' to QUIT")

    while True:
        choice = input("\n👉 Enter command: ").strip().lower()

        if choice == 'q':
            print("Exiting...")
            break
        
        elif choice == 's':
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                print("Music stopped.")
            else:
                print("Nothing is playing.")

        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(recommendations):
                song_name = recommendations[idx][0]
                song_path = recommendations[idx][2]

                # Validate file existence
                if os.path.exists(song_path):
                    try:
                        pygame.mixer.music.load(song_path)
                        pygame.mixer.music.play()
                        print(f"NOW PLAYING: {song_name}")
                    except Exception as e:
                        print(f"Error playing file: {e}")
                else:
                    print(f"File not found: {song_path}")
            else:
                print("Invalid number. Please select from the list.")
        else:
            print(" Invalid command.")

if __name__ == "__main__":
    main_test()