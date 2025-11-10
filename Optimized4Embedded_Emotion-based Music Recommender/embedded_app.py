import numpy as np
import cv2
import sqlite3 # Use SQLite instead of Pandas
import os
import time
import psutil # For benchmarking
from collections import Counter
from datetime import datetime
import glob # Added for test image loading

# Use tflite_runtime for inference
import tflite_runtime.interpreter as tflite

# Use Pygame for basic UI AND audio
import pygame

# --- Configuration ---
# Use the Float32 model for better accuracy
TFLITE_MODEL_PATH = 'op_model_float32.tflite' # <-- Make sure you're using the Float32 model
SQLITE_DB_PATH = 'music.db'
# Copy the .xml file to your project folder
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml' 
DB_TABLE_NAME = 'songs'
TEST_IMAGE_FOLDER = 'test_images' # Folder with your 3 test images

# --- Screen dimensions for Pygame ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAMERA_FEED_POS = (50, 50)
CAMERA_FEED_SIZE = (480, 360) 
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
    Returns a list of tuples: (name, artist, link)
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
                SELECT name, artist, link
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

# --- Main Application ---
def main():
    # --- Initialization ---
    pygame.init()
    pygame.mixer.init() # <-- INITIALIZE THE AUDIO MIXER
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Emotion-Based Music Recommender')
    font = pygame.font.Font(None, FONT_SIZE)
    clock = pygame.time.Clock()
    
    # --- Load TFLite Model ---
    try:
        print(f"Loading TFLite model from: {TFLITE_MODEL_PATH}")
        # Make sure you're using the Float32 model for accuracy
        interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH) 
        interpreter.allocate_tensors()
        print("✅ TFLite model loaded and tensors allocated.")
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        input_type = input_details[0]['dtype']
        print(f"Model expects input shape: ({height}, {width}), type: {input_type}")
        print(f"Model output shape: {output_details[0]['shape']}, type: {output_details[0]['dtype']}")

    except Exception as e:
        print(f"❌ Failed to load TFLite model: {e}")
        return

    # --- Load Haar Cascade ---
    try:
        print(f"Loading Haarcascade Classifier from: {HAAR_CASCADE_PATH}")
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade.empty():
            raise IOError(f"Cannot load Haarcascade classifier from {HAAR_CASCADE_PATH}")
        print("✅ Haarcascade Classifier loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load Haarcascade: {e}")
        return

    # --- Application State ---
    running = True
    scanning = False
    detected_emotions_list = []
    recommendations = []
    benchmark_results = {}
    last_scan_time = 0
    scan_duration = 3 # seconds to scan
    song_rects = [] # To store clickable areas for songs
    currently_playing_song = ""
    cap = None # Camera object

    # --- Main Loop ---
    while running:
        # --- Event Handling ---
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # --- CLICK HANDLING FOR AUDIO ---
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left click
                for i, rect in enumerate(song_rects):
                    if rect.collidepoint(mouse_pos):
                        if i < len(recommendations):
                            song_filepath = recommendations[i][2] # Get the filepath
                            song_name = recommendations[i][0]
                            
                            if not os.path.exists(song_filepath):
                                print(f"❌ Error: File not found! {song_filepath}")
                                currently_playing_song = "Error: File not found"
                                break

                            print(f"Playing: {song_filepath}")
                            try:
                                pygame.mixer.music.load(song_filepath)
                                pygame.mixer.music.play()
                                currently_playing_song = f"Playing: {song_name[:35]}" # Truncate
                            except Exception as e:
                                print(f"Failed to play {song_filepath}: {e}")
                                currently_playing_song = "Error playing file"
            # --- END CLICK HANDLING ---

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s: # 's' to stop music
                    pygame.mixer.music.stop()
                    print("Music stopped.")
                    currently_playing_song = ""
                
                if event.key == pygame.K_SPACE and not scanning:
                    print("\n--- Starting Emotion Scan ---")
                    scanning = True
                    detected_emotions_list.clear()
                    recommendations.clear()
                    benchmark_results.clear()
                    song_rects.clear()
                    currently_playing_song = ""
                    last_scan_time = time.time()
                    
                    # === BENCHMARK START ===
                    process = psutil.Process(os.getpid())
                    start_mem_scan = process.memory_info().rss / (1024 * 1024)
                    total_inference_time_scan = 0
                    num_inferences_scan = 0
                    # === BENCHMARK START ===

                    cap = cv2.VideoCapture(0) # Open camera
                    if not cap.isOpened():
                        print("❌ Error: Cannot open camera.")
                        scanning = False # Abort scan


        # --- Screen Drawing ---
        screen.fill((30, 30, 30)) 
        song_rects.clear() 

        # --- Camera Feed and Processing (if scanning) ---
        if scanning:
            current_time = time.time()
            # --- Check if scan duration is over ---
            if current_time - last_scan_time > scan_duration:
                print(f"--- Scan finished after {scan_duration} seconds ---")
                scanning = False
                if cap and cap.isOpened():
                    cap.release()
                
                # === BENCHMARK END ===
                end_mem_scan = process.memory_info().rss / (1024 * 1024)
                total_time_scan = current_time - last_scan_time
                avg_inference_scan = (total_inference_time_scan / num_inferences_scan) * 1000 if num_inferences_scan else 0
                fps_scan = num_inferences_scan / total_time_scan if total_time_scan > 0 else 0
                mem_used_scan = end_mem_scan - start_mem_scan

                benchmark_results = {
                    "avg_inference_ms": avg_inference_scan,
                    "fps": fps_scan,
                    "mem_used_MB": mem_used_scan,
                    "frames_processed": num_inferences_scan
                }
                print("Benchmark Results:", benchmark_results)

                # ... (Log results - same as your test script) ...
                try:
                    log_entry_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "avg_inference_ms": benchmark_results.get('avg_inference_ms', 0),
                        "fps": benchmark_results.get('fps', 0),
                        "mem_used_MB": benchmark_results.get('mem_used_MB', 0),
                        "frames": benchmark_results.get('frames_processed', 0)
                    }
                    log_entry = [log_entry_data]
                    log_file = "benchmark_log.csv"
                    file_exists = os.path.isfile(log_file)
                    with open(log_file, 'a', newline='') as f:
                        import csv
                        fieldnames = log_entry_data.keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                        writer.writerows(log_entry)
                    print("✅ Benchmark data recorded to benchmark_log.csv")
                except Exception as e:
                    print(f"⚠️ Could not save benchmark log: {e}")
                # === BENCHMARK END ===

                # Process emotions and get recommendations
                processed_emotions = pre(detected_emotions_list)
                if processed_emotions:
                    recommendations = get_recommendations_from_db(processed_emotions)
                else:
                    print("No emotions detected during scan.")

            # --- Process frame if still scanning ---
            elif cap and cap.isOpened():
                ret, frame = cap.read()
                display_frame = frame.copy() 

                if ret:
                    # --- Pre-processing ---
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

                    # --- Inference ---
                    for (x, y, w, h) in faces:
                        roi_gray = gray[y:y + h, x:x + w]
                        try:
                            # --- Pre-processing for TFLite ---
                            img_resized = cv2.resize(roi_gray, (width, height))
                            img_expanded_channel = np.expand_dims(img_resized, axis=-1)
                            # Assuming Float32 input based on our last check
                            input_data = img_expanded_channel.astype(np.float32) / 255.0
                            input_data = np.expand_dims(input_data, axis=0) 

                            # --- TFLite Inference ---
                            t0 = time.time()
                            interpreter.set_tensor(input_details[0]['index'], input_data)
                            interpreter.invoke()
                            output_data = interpreter.get_tensor(output_details[0]['index'])
                            t1 = time.time()
                            total_inference_time_scan += (t1 - t0)
                            num_inferences_scan += 1
                            # --- TFLite Inference ---

                            # --- Post-processing ---
                            probabilities = output_data[0] # Assuming float32 output
                            max_index = int(np.argmax(probabilities))
                            detected_emotion = emotion_dict.get(max_index, "Unknown")
                            detected_emotions_list.append(detected_emotion)

                            # --- Draw on display frame ---
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(display_frame, detected_emotion, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        except Exception as e:
                            print(f"Error during face processing/inference: {e}")

                    # --- Display Camera Feed via Pygame ---
                    display_frame_resized = cv2.resize(display_frame, CAMERA_FEED_SIZE)
                    display_frame_rgb = cv2.cvtColor(display_frame_resized, cv2.COLOR_BGR2RGB)
                    display_frame_rgb = np.rot90(display_frame_rgb)
                    pygame_surface = pygame.surfarray.make_surface(display_frame_rgb)
                    screen.blit(pygame_surface, CAMERA_FEED_POS)
                    scan_time_left = max(0, scan_duration - (time.time() - last_scan_time))
                    draw_text(screen, f"Scanning... {scan_time_left:.1f}s", (CAMERA_FEED_POS[0], CAMERA_FEED_POS[1] + CAMERA_FEED_SIZE[1] + 5), font, (255, 255, 0))

                else:
                    draw_text(screen, "Error reading camera frame.", (CAMERA_FEED_POS[0], CAMERA_FEED_POS[1]), font, (255, 0, 0))
            
            elif not (cap and cap.isOpened()):
                 draw_text(screen, "Camera not available.", (CAMERA_FEED_POS[0], CAMERA_FEED_POS[1]), font, (255, 0, 0))
                 scanning = False # Reset flag

        # --- Display Idle Message / Instructions ---
        else:
            draw_text(screen, "Press SPACEBAR to Scan Emotion", (CAMERA_FEED_POS[0], CAMERA_FEED_POS[1]), font, (0, 255, 255))
            draw_text(screen, "Click a song to play. Press 'S' to stop.", (CAMERA_FEED_POS[0], CAMERA_FEED_POS[1] + 20), font, (0, 255, 255))

        # --- Display Recommendations (and create clickable Rects) ---
        if recommendations:
            draw_text(screen, "Recommendations (Click to play):", RECOMMENDATION_POS, font, (0, 255, 0))
            y_offset = FONT_SIZE + 5
            for i, (name, artist, link) in enumerate(recommendations):
                if i >= 25: break 
                rec_text = f"{i+1}. {name[:30]} - {artist[:25]}"
                
                text_color = (255, 255, 255) # Default white
                # Create a Rect for the text
                text_rect = pygame.Rect(RECOMMENDATION_POS[0], RECOMMENDATION_POS[1] + y_offset, SCREEN_WIDTH - RECOMMENDATION_POS[0] - 20, FONT_SIZE + 2)
                
                if text_rect.collidepoint(mouse_pos):
                    text_color = (255, 255, 0) # Highlight in yellow
                
                draw_text(screen, rec_text, text_rect.topleft, font, text_color)
                song_rects.append(text_rect) 
                y_offset += FONT_SIZE + 2

        # --- Display Currently Playing Song ---
        if currently_playing_song:
            draw_text(screen, currently_playing_song, (RECOMMENDATION_POS[0], SCREEN_HEIGHT - 40), font, (0, 255, 255))

        # --- Display Benchmark Results ---
        if benchmark_results:
            draw_text(screen, "Last Scan Performance:", BENCHMARK_POS, font, (255, 255, 0))
            y_offset = FONT_SIZE + 5
            draw_text(screen, f"Avg Inference: {benchmark_results.get('avg_inference_ms', 0):.2f} ms", (BENCHMARK_POS[0], BENCHMARK_POS[1] + y_offset), font)
            y_offset += FONT_SIZE + 2
            draw_text(screen, f"Approx FPS: {benchmark_results.get('fps', 0):.2f}", (BENCHMARK_POS[0], BENCHMARK_POS[1] + y_offset), font)
            y_offset += FONT_SIZE + 2
            draw_text(screen, f"Memory Used: {benchmark_results.get('mem_used_MB', 0):.2f} MB", (BENCHMARK_POS[0], BENCHMARK_POS[1] + y_offset), font)
            y_offset += FONT_SIZE + 2
            draw_text(screen, f"Frames Processed: {benchmark_results.get('frames_processed', 0)}", (BENCHMARK_POS[0], BENCHMARK_POS[1] + y_offset), font)


        # --- Update Display ---
        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    # --- Cleanup ---
    if cap and cap.isOpened():
        cap.release()
    pygame.mixer.quit() # <-- QUIT THE MIXER
    pygame.quit()
    print("Application exited.")


if __name__ == "__main__":
    main()