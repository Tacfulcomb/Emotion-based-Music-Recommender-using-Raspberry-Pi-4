import numpy as np
import cv2
import sqlite3 
import os
import time
import psutil # For benchmarking
from collections import Counter
from datetime import datetime
import glob 
import csv
import mediapipe as mp

# Use tflite_runtime for inference
import tflite_runtime.interpreter as tflite

# Use Pygame for basic UI AND audio
import pygame

# --- Configuration ---
# Use the Float32 model for better accuracy
TFLITE_MODEL_PATH = 'op_model_float32.tflite' 
SQLITE_DB_PATH = 'music.db'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml' 
DB_TABLE_NAME = 'songs'
TEST_IMAGE_FOLDER = 'test_images' 
DETECTOR_MODE = 'HAARCASCADE' 
# --- Screen dimensions for Pygame ---
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
CAMERA_FEED_POS = (50, 50)
CAMERA_FEED_SIZE = (1920, 1080) 
RECOMMENDATION_POS = (550, 50)
BENCHMARK_POS = (50, 450)
FONT_SIZE = 18

# --- Emotion Mapping (from original script) ---
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# --- Helper Function for Pygame Text ---
def draw_text(surface, text, pos, font, color=(255, 255, 255)):
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos)

# --- Refactored function to get songs from SQLite ---
def get_recommendations_from_db(emotion_list):
    """
    Queries the SQLite database for random songs based on the detected emotions.
    Returns a list of tuples: (name,link)
    """
    recommendations = []
    conn = None 
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        print(f"Querying recommendations for emotions: {emotion_list}")

        # Determine number of songs per emotion
        num_emotions = len(emotion_list)
        if num_emotions == 1:
            times = [10] # Get 10 songs
        elif num_emotions == 2:
            times = [7, 3] # 7 + 3
        else: # 3 or more
            times = [5, 3, 2] # 5 + 3 + 2 (and ignore others)

        for i, emotion in enumerate(emotion_list):
            limit = times[i] if i < len(times) else 0 # Stop after the defined list
            if limit <= 0: continue 

            query = f"""
                SELECT name, link
                FROM {DB_TABLE_NAME}
                WHERE emotion_category = ? 
                AND link IS NOT NULL AND link != '' 
                ORDER BY RANDOM()
                LIMIT ?
            """
            # Added "AND link IS NOT NULL..." to ensure we only get songs with a valid file path
            
            cursor.execute(query, (emotion, limit))
            songs = cursor.fetchall()
            print(f" - Found {len(songs)} songs with file paths for {emotion} (limit {limit})")
            recommendations.extend(songs) 

        conn.close()
        print(f"Total recommendations fetched: {len(recommendations)}")
        np.random.shuffle(recommendations)
        return recommendations[:10] # Return max 10 recommendations

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        if conn: conn.close()
        return [] 
    except Exception as e:
        print(f"❌ Error in get_recommendations_from_db: {e}")
        if conn: conn.close()
        return []


# --- Emotion List Processing (kept from original) ---
def pre(l):
    """Processes raw emotion list to get unique, frequency-sorted list."""
    if not l:
        return []
    try:
        emotion_counts = Counter(l)
        sorted_emotions = sorted(emotion_counts.items(), key=lambda item: (-item[1], item[0]))
        ul = [emotion for emotion, count in sorted_emotions]
        print(f"Processed unique emotions (sorted by frequency): {ul}")
        return ul
    except Exception as e:
        print(f"Error in pre function: {e}")
        return []

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


def main():
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f'Benchmark Mode: {DETECTOR_MODE}')
    font = pygame.font.Font(None, FONT_SIZE)
    clock = pygame.time.Clock()
    
    # --- Load Detectors ---
    mp_face = mp.solutions.face_detection
    face_detector_mp = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    # --- Load TFLite Model ---
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH) 
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]

    # --- Application State ---
    running, scanning = True, False
    detected_emotions_data = []
    recommendations = []
    benchmark_results = {}
    last_scan_time = 0
    scan_duration = 3 
    cap = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not scanning:
                    scanning = True
                    detected_emotions_data.clear()
                    recommendations.clear()
                    last_scan_time = time.time()
                    process = psutil.Process(os.getpid())
                    start_mem = process.memory_info().rss / (1024 * 1024)
                    total_inf_time, num_inf = 0, 0
                    cap = cv2.VideoCapture(0)

        screen.fill((30, 30, 30))
        # --- UI DRAWING (RECOMMENDATIONS) ---
        draw_text(screen, "Recommendations:", RECOMMENDATION_POS, font, (0, 255, 0))
        
        if recommendations:
            y_offset = 30 # Initial vertical gap from header
            for i, (name, link) in enumerate(recommendations):
                # Ensure it fits on screen
                if i >= 10: break 
                
                rec_text = f"{i+1}. {name[:40]}"
                draw_text(screen, rec_text, (RECOMMENDATION_POS[0], RECOMMENDATION_POS[1] + y_offset), font)
                y_offset += 25 # Space between song titles
        elif not scanning:
            # Message when idle
            draw_text(screen, "Press SPACE to Scan for Music", (RECOMMENDATION_POS[0], RECOMMENDATION_POS[1] + 30), font, (150, 150, 150))
        if scanning:
            current_time = time.time()
            if current_time - last_scan_time > scan_duration:
                # === FINISH SCAN ===
                scanning = False
                if cap: cap.release()
                
                # Calculate Final Stats
                avg_inf = (total_inf_time / num_inf) * 1000 if num_inf else 0
                final_emotion = "None"
                final_conf = 0.0
                
                if detected_emotions_data:
                    # Get most frequent emotion
                    counts = Counter([e[0] for e in detected_emotions_data])
                    final_emotion = counts.most_common(1)[0][0]
                    # Avg confidence for that specific emotion
                    final_conf = np.mean([e[1] for e in detected_emotions_data if e[0] == final_emotion])

                benchmark_results = {
                    "detector": DETECTOR_MODE,
                    "emotion": final_emotion,
                    "confidence": final_conf,
                    "avg_inf_ms": avg_inf,
                    "fps": num_inf / scan_duration
                }

                # === LOG TO CSV ===
                with open("benchmark_log_sync.csv", 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=["timestamp", "detector", "emotion", "confidence_pct", "avg_inf_ms", "fps"])
                    if not os.path.isfile("benchmark_log_sync.csv.csv"): writer.writeheader()
                    writer.writerow({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "detector": DETECTOR_MODE,
                        "emotion": final_emotion,
                        "confidence_pct": f"{final_conf*100:.2f}%",
                        "avg_inf_ms": f"{avg_inf:.2f}",
                        "fps": f"{benchmark_results['fps']:.2f}"
                    })

                processed_names = [e[0] for e in detected_emotions_data]
                recommendations = get_recommendations_from_db(pre(processed_names))

            elif cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    display_frame = frame.copy()
                    roi_gray, coords = None, None
                    
                    # --- DETECTION PHASE ---
                    if DETECTOR_MODE == 'MEDIAPIPE':
                        results = face_detector_mp.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if results.detections:
                            b = results.detections[0].location_data.relative_bounding_box
                            ih, iw, _ = frame.shape
                            x, y, w, h = int(b.xmin*iw), int(b.ymin*ih), int(b.width*iw), int(b.height*ih)
                            coords = (x, y, w, h)
                    else:
                        faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)
                        if len(faces) > 0: coords = faces[0]

                    # --- INFERENCE PHASE ---
                    if coords is not None:
                        x, y, w, h = coords
                        x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1]-x), min(h, frame.shape[0]-y)
                        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                        
                        if roi_gray.size > 0:
                            blob = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (width, height)).astype(np.float32) / 255.0, axis=-1), axis=0)
                            t0 = time.time()
                            interpreter.set_tensor(input_details[0]['index'], blob)
                            interpreter.invoke()
                            output = interpreter.get_tensor(output_details[0]['index'])[0]
                            total_inf_time += (time.time() - t0)
                            num_inf += 1
                            
                            idx = np.argmax(output)
                            conf = output[idx]
                            label = emotion_dict[idx]
                            detected_emotions_data.append((label, float(conf)))
                            
                            # UI Feedback
                            color = (255, 0, 255) if DETECTOR_MODE == 'MEDIAPIPE' else (0, 255, 0)
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(display_frame, f"{label} {conf*100:.0f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Draw Camera to Pygame
                    surf = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(cv2.resize(display_frame, CAMERA_FEED_SIZE), cv2.COLOR_BGR2RGB)))
                    screen.blit(surf, CAMERA_FEED_POS)

        # --- UI DRAWING (STATS) ---
        if benchmark_results:
            y = BENCHMARK_POS[1]
            draw_text(screen, f"Detector: {benchmark_results['detector']}", (BENCHMARK_POS[0], y), font, (255, 255, 0))
            draw_text(screen, f"Emotion: {benchmark_results['emotion']} ({benchmark_results['confidence']*100:.1f}%)", (BENCHMARK_POS[0], y+25), font)
            draw_text(screen, f"Inference: {benchmark_results['avg_inf_ms']:.2f} ms", (BENCHMARK_POS[0], y+50), font)
            draw_text(screen, f"FPS: {benchmark_results['fps']:.2f}", (BENCHMARK_POS[0], y+75), font)

        pygame.display.flip()
        clock.tick(60)

    if cap: cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
