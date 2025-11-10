import numpy as np
import cv2
import sqlite3 # Use SQLite instead of Pandas
import os
import time
import psutil # For benchmarking
from collections import Counter
from datetime import datetime
import glob # To find image files

# Use tflite_runtime for inference
import tflite_runtime.interpreter as tflite

# --- Configuration ---
TFLITE_MODEL_PATH = 'op_model.tflite'
SQLITE_DB_PATH = 'music.db'
# Adjust path if you copied the xml file to the current directory
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
DB_TABLE_NAME = 'songs'
TEST_IMAGE_FOLDER = 'test_images' # Folder containing your 3 test images

# --- Emotion Mapping (from original script) ---
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

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
        print(f"\n--- Querying recommendations for emotions: {emotion_list} ---")

        num_emotions = len(emotion_list)
        # Simplified recommendation count for testing
        if num_emotions == 1:
            times = [10] # Get 10 songs for one emotion
        elif num_emotions == 2:
            times = [7, 3] # Get 7+3 songs
        else: # 3 emotions
            times = [5, 3, 2] # Get 5+3+2 songs

        for i, emotion in enumerate(emotion_list):
            limit = times[i] if i < len(times) else times[-1]
            if limit <= 0: continue

            print(f"   Querying {limit} songs for '{emotion}'...")
            query = f"""
                SELECT name, artist, link
                FROM {DB_TABLE_NAME}
                WHERE emotion_category = ?
                ORDER BY RANDOM()
                LIMIT ?
            """
            cursor.execute(query, (emotion, limit))
            songs = cursor.fetchall()
            print(f"   -> Found {len(songs)} songs for '{emotion}'")
            recommendations.extend(songs)

        conn.close()
        print(f"Total recommendations fetched: {len(recommendations)}")
        np.random.shuffle(recommendations)
        # Limit total recommendations if needed, e.g., return recommendations[:10]
        return recommendations

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
        print("\n--- Processing detected emotions ---")
        print(f"Raw list: {l}")
        emotion_counts = Counter(l)
        sorted_emotions = sorted(emotion_counts.items(), key=lambda item: (-item[1], item[0]))
        ul = [emotion for emotion, count in sorted_emotions]
        print(f"Processed unique emotions (sorted by frequency): {ul}")
        return ul
    except Exception as e:
        print(f"Error in pre function: {e}")
        return []

# --- Main Test Function ---
def main_test():
    # --- Initialization ---
    print("--- Test Script Initializing ---")

    # --- Load TFLite Model ---
    try:
        print(f"Loading TFLite model from: {TFLITE_MODEL_PATH}")
        interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        print("✅ TFLite model loaded and tensors allocated.")

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # Keep details brief for test output
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        input_type = input_details[0]['dtype']
        print(f"Model expects input shape: (1, {height}, {width}, 1), type: {input_type}")
        print(f"Model output shape: {output_details[0]['shape']}, type: {output_details[0]['dtype']}")


    except Exception as e:
        print(f"❌ Failed to load TFLite model: {e}")
        return

    # --- Load Haar Cascade ---
    try:
        print(f"Loading Haarcascade Classifier from: {HAAR_CASCADE_PATH}")
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade.empty():
            raise IOError("Cannot load Haarcascade classifier")
        print("✅ Haarcascade Classifier loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load Haarcascade: {e}")
        return

    # --- Application State ---
    detected_emotions_list = []
    recommendations = []
    benchmark_results = {}

    # --- Find Test Images ---
    image_patterns = [os.path.join(TEST_IMAGE_FOLDER, '*.png'),
                      os.path.join(TEST_IMAGE_FOLDER, '*.jpg'),
                      os.path.join(TEST_IMAGE_FOLDER, '*.jpeg')]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern))

    if not image_files:
        print(f"❌ Error: No test images found in '{TEST_IMAGE_FOLDER}' folder.")
        print(f"   Searched in: {os.path.abspath(TEST_IMAGE_FOLDER)}")
        return

    print(f"\n--- Starting Emotion Detection on {len(image_files)} test images ---")

    # === BENCHMARK START ===
    process = psutil.Process(os.getpid())
    start_mem_scan = process.memory_info().rss / (1024 * 1024)
    start_time_scan = time.time()
    total_inference_time_scan = 0
    num_inferences_scan = 0
    # === BENCHMARK START ===

    # --- Loop through Test Images ---
    for image_path in image_files:
        print(f"\nProcessing image: {os.path.basename(image_path)}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read {image_path}. Skipping.")
            continue

        # --- Pre-processing ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use slightly gentler parameters for static images if needed
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        if len(faces) == 0:
            print("   -> No face detected.")
            continue
        else:
             print(f"   -> Detected {len(faces)} face(s). Processing the first one.")

        # --- Inference (process first detected face) ---
        x, y, w, h = faces[0] # Get coordinates of the first face
        roi_gray = gray[y:y + h, x:x + w]
        try:
            # Resize and prepare input tensor
            img_resized = cv2.resize(roi_gray, (width, height))
            img_expanded_channel = np.expand_dims(img_resized, axis=-1)

            if input_type == np.float32:
                input_data = img_expanded_channel.astype(np.float32) / 255.0
            # Add elif for uint8/int8 if needed based on inspect_model.py output
            else: # Fallback assumption
                 input_data = img_expanded_channel.astype(np.float32) / 255.0

            input_data = np.expand_dims(input_data, axis=0) # Add batch dimension

            # --- TFLite Inference ---
            t0 = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke() # Run inference
            output_data = interpreter.get_tensor(output_details[0]['index'])
            t1 = time.time()
            inference_time_ms = (t1 - t0) * 1000
            total_inference_time_scan += (t1 - t0)
            num_inferences_scan += 1
            print(f"   -> Inference time: {inference_time_ms:.2f} ms")
            # --- TFLite Inference ---

            # --- Post-processing ---
            probabilities = output_data[0] # Assuming float32 output [1, 7]
            max_index = int(np.argmax(probabilities))
            detected_emotion = emotion_dict.get(max_index, "Unknown")
            confidence = probabilities[max_index]
            detected_emotions_list.append(detected_emotion)
            print(f"   -> Detected emotion: {detected_emotion} (Confidence: {confidence:.2f})")

        except Exception as e:
            print(f"   -> Error during face processing/inference: {e}")
            import traceback
            traceback.print_exc()

    # === BENCHMARK END ===
    end_time_scan = time.time()
    end_mem_scan = process.memory_info().rss / (1024 * 1024)
    total_time_scan = end_time_scan - start_time_scan
    avg_inference_scan = (total_inference_time_scan / num_inferences_scan) * 1000 if num_inferences_scan else 0
    # FPS based on total processing time for all images, less meaningful here than avg inference
    # fps_scan = num_inferences_scan / total_time_scan if total_time_scan > 0 else 0
    mem_used_scan = end_mem_scan - start_mem_scan

    benchmark_results = {
        "avg_inference_ms_per_face": avg_inference_scan,
        # "fps_overall": fps_scan, # Less relevant for static images
        "memory_used_MB_scan": mem_used_scan,
        "faces_processed": num_inferences_scan,
        "total_processing_time_s": total_time_scan
    }
    print("\n--- Image Processing Finished ---")

    # Log results
    try:
        log_entry_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "avg_inference_ms": benchmark_results.get('avg_inference_ms_per_face', 0),
            # "fps": benchmark_results.get('fps_overall', 0),
            "mem_used_MB": benchmark_results.get('memory_used_MB_scan', 0),
            "frames": benchmark_results.get('faces_processed', 0),
            "total_time_s": benchmark_results.get('total_processing_time_s', 0)
        }
        log_entry = [log_entry_data]
        log_file = "benchmark_log.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as f:
            import csv
            # Adjust fieldnames dynamically based on keys
            fieldnames = log_entry_data.keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(log_entry)
        print("✅ Benchmark data recorded to benchmark_log.csv")
    except Exception as e:
        print(f"⚠️ Could not save benchmark log: {e}")
    # === BENCHMARK END ===

    # --- Process emotions and get recommendations ---
    processed_emotions = pre(detected_emotions_list)
    if processed_emotions:
        recommendations = get_recommendations_from_db(processed_emotions)
    else:
        print("\nNo valid emotions detected from test images to generate recommendations.")

    # --- Print Recommendations (Simulating LCD) ---
    print("\n--- Recommendations (Simulated LCD Output) ---")
    if recommendations:
        # Limit to simulate display space
        display_limit = 10
        for i, (name, artist, link) in enumerate(recommendations[:display_limit]):
             # Simple print mimics LCD line output
             print(f"{i+1:02d}: {name[:20]} - {artist[:15]}") # Truncate names
    else:
        print("   No recommendations generated.")

    # --- Print Benchmark Results (Simulating LCD) ---
    print("\n--- Benchmark Results (Simulated LCD Output) ---")
    if benchmark_results:
         print(f"Avg Inference: {benchmark_results.get('avg_inference_ms_per_face', 0):.1f} ms")
         # print(f"Overall FPS: {benchmark_results.get('fps_overall', 0):.1f}")
         print(f"Memory Used:   {benchmark_results.get('memory_used_MB_scan', 0):.1f} MB")
         print(f"Faces Found:   {benchmark_results.get('faces_processed', 0)}")
         print(f"Total Time:    {benchmark_results.get('total_processing_time_s', 0):.2f} s")

    print("\n--- Test Script Finished ---")


if __name__ == "__main__":
    main_test()