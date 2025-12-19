import numpy as np
import cv2
import sqlite3
import os
import time
import threading
import queue
import csv
import psutil
from collections import Counter
from datetime import datetime
import tflite_runtime.interpreter as tflite
import pygame
import mediapipe as mp

# --- Configuration ---
CONFIG = {
    'MODEL_PATH': 'op_model_float32.tflite',
    'DB_PATH': 'music.db',
    'HAAR_PATH': 'haarcascade_frontalface_default.xml',
    'TABLE_NAME': 'songs',
    'DETECTOR_TYPE': 'HAARCASCADE',
    'SCREEN_W': 1920,
    'SCREEN_H': 1080,
    'CAM_SIZE': (960, 720),
    'EMOTIONS': {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
}
# --- Layout Offsets ---
LAYOUT = {
    'CAM_X': 50,
    'CAM_Y': 50,
    'SIDEBAR_X_OFFSET': 1350, # How far from the left the playlist starts
    'STATS_Y_OFFSET': 820,     # How far from the top the benchmark results start
    'LINE_SPACING': 35         # Vertical gap between song titles
}

# --- Profiler ---
class Profiler:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()
    def reset(self):
        with self.lock:
            self.cam_times = []
            self.pre_times = []
            self.inf_times = []
            self.db_time = 0.0
    def record(self, stage, ms):
        with self.lock:
            if stage == 'cam': self.cam_times.append(ms)
            elif stage == 'pre': self.pre_times.append(ms)
            elif stage == 'inf': self.inf_times.append(ms)
            elif stage == 'db': self.db_time = ms
    def get_averages(self):
        with self.lock:
            def avg(l): return sum(l) / len(l) if l else 0.0
            return {
                'avg_cam_ms': avg(self.cam_times),
                'avg_pre_ms': avg(self.pre_times),
                'avg_inf_ms': avg(self.inf_times),
                'total_db_ms': self.db_time,
                'frames_processed': len(self.inf_times)
            }

profiler = Profiler()

# --- Module 1: Music Controller (STRICT MODE) ---
class MusicController:
    def __init__(self, db_path):
        self.db_path = db_path
        pygame.mixer.init()
        self.SONG_END = pygame.USEREVENT + 1
        pygame.mixer.music.set_endevent(self.SONG_END)

    def is_playing(self):
        return pygame.mixer.music.get_busy()

    def get_recommendations(self, emotion_list):
        if not emotion_list: return []
        t0 = time.time()
        
        # 1. Find the Winner (Strict Mode)
        # We only care about the #1 most common emotion
        counts = Counter(emotion_list)
        try:
            top_emotion, _ = counts.most_common(1)[0]
        except IndexError:
            return []

        # Safety Check
        if top_emotion == "None" or top_emotion == "Unknown":
            return []

        recommendations = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 2. STRICT QUERY: Only get songs for the Winner (LIMIT 10)
            query = f"SELECT name, link FROM {CONFIG['TABLE_NAME']} WHERE emotion_category = ? AND link IS NOT NULL ORDER BY RANDOM() LIMIT 10"
            
            # Tuple syntax is critical here
            cursor.execute(query, (top_emotion,)) 
            
            recommendations = cursor.fetchall()
            conn.close()
            
            t1 = time.time()
            profiler.record('db', (t1 - t0) * 1000)
            
            np.random.shuffle(recommendations)
            return recommendations
            
        except Exception as e:
            print(f"❌ DB Error: {e}")
            return []
    
    def play_song(self, filepath):
        if not os.path.exists(filepath): return "Error: File not found"
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            return f"Playing: {os.path.basename(filepath)[:30]}..."
        except Exception as e:
            return "Error playing file"
    
    def stop(self):
        pygame.mixer.music.stop()

# --- Module 2: Camera Thread ---
class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.running = True
        self.scanning = False
        self.lock = threading.Lock()
    def run(self):
        while self.running and self.cap.isOpened():
            t0 = time.time()
            ret, img = self.cap.read()
            t1 = time.time()
            if ret:
                with self.lock: self.frame = img
                if self.scanning: profiler.record('cam', (t1 - t0) * 1000)
            else: time.sleep(0.01)
    def start_scan(self): self.scanning = True
    def stop_scan(self): self.scanning = False
    def get_frame(self):
        with self.lock: return self.frame.copy() if self.frame is not None else None
    def stop(self):
        self.running = False
        self.cap.release()

# --- Module 3: Inference Thread ---
class InferenceThread(threading.Thread):
    def __init__(self, camera_thread):
        super().__init__()
        self.cam = camera_thread
        self.running = True
        self.scanning = False 
        self.latest_result = []
        self.lock = threading.Lock()
        
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(min_detection_confidence=0.5)
        
        try:
            self.interpreter = tflite.Interpreter(model_path=CONFIG['MODEL_PATH'])
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("? Inference Engine Loaded")
        except Exception as e:
            print(f"? ML Load Error: {e}")
            self.running = False

    def start_scan(self):
        self.scanning = True

    def stop_scan(self):
        self.scanning = False

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            if not self.scanning:
                time.sleep(0.01)
                continue
            
            frame = self.cam.get_frame()
            if frame is None: continue
            
            t_pre_start = time.time()
            emotions_found = []
            processed_roi = None

            # 1. PROCESS frame first to get results
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_frame)
            
            # 2. NOW check for detections
            if results.detections:
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                w, h = int(bboxC.width * iw), int(bboxC.height * ih)

                # ROI Extraction
                x, y = max(0, x), max(0, y)
                w, h = min(w, iw - x), min(h, ih - y)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi = gray[y:y+h, x:x+w]

                if roi.size > 0:
                    roi = cv2.resize(roi, (48, 48))
                    roi = roi.astype(np.float32) / 255.0
                    processed_roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

            t_pre_end = time.time()
            profiler.record('pre', (t_pre_end - t_pre_start) * 1000)

            if processed_roi is not None:
                t_inf_start = time.time()
                self.interpreter.set_tensor(self.input_details[0]['index'], processed_roi)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                t_inf_end = time.time()
                profiler.record('inf', (t_inf_end - t_inf_start) * 1000)
                
                max_idx = np.argmax(output)
                emotions_found.append((CONFIG['EMOTIONS'][max_idx], float(output[max_idx])))
                
            with self.lock:
                self.latest_result = emotions_found

    def get_results(self):
        with self.lock: return list(self.latest_result)
# --- Module 4: Main UI Application ---
class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((CONFIG['SCREEN_W'], CONFIG['SCREEN_H']))
        pygame.display.set_caption("Co-Design Benchmark Tool + Vibe Mode")
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 40)
        self.clock = pygame.time.Clock()

        self.music_ctrl = MusicController(CONFIG['DB_PATH'])
        self.cam_thread = CameraThread()
        self.inf_thread = InferenceThread(self.cam_thread)
        
        self.running = True
        self.state = "IDLE" 
        self.scan_start_time = 0
        self.scan_duration = 3.0
        self.collected_emotions = [] # Stores tuples (label, score)
        self.recommendations = []
        self.song_rects = []
        self.final_stats = {} 
        self.start_mem = 0
        self.status_msg = "Ready"

    def run(self):
        self.cam_thread.start()
        self.inf_thread.start()
        while self.running:
            self.handle_input()
            self.update_logic()
            self.draw()
            self.clock.tick(30)
        self.cam_thread.stop()
        self.inf_thread.stop()
        self.cam_thread.join()
        self.inf_thread.join()
        pygame.quit()

    def handle_input(self):
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            
            if event.type == self.music_ctrl.SONG_END:
                if self.state == "VIBE_PLAY":
                    print("🎵 Song ended. Restarting Vibe Scan...")
                    self.start_scan(vibe_mode=True)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    print("Quitting Application...")
                    self.running = False
                
                if event.key == pygame.K_SPACE and self.state == "IDLE":
                    self.start_scan(vibe_mode=False)
                
                elif event.key == pygame.K_v:
                    if "VIBE" in self.state:
                        self.state = "IDLE"
                        self.music_ctrl.stop()
                        self.inf_thread.stop_scan()
                        self.cam_thread.stop_scan()
                        self.status_msg = "Vibe Mode OFF"
                    else:
                        self.start_scan(vibe_mode=True)

                elif event.key == pygame.K_s:
                    self.music_ctrl.stop()
                    self.status_msg = "Music Stopped"
                    if "VIBE" in self.state: self.state = "IDLE" 

            if event.type == pygame.MOUSEBUTTONDOWN:
                for idx, rect in enumerate(self.song_rects):
                    if rect.collidepoint(mouse_pos) and idx < len(self.recommendations):
                        name, link = self.recommendations[idx]
                        self.status_msg = self.music_ctrl.play_song(link)

    def start_scan(self, vibe_mode=False):
        if vibe_mode:
            self.state = "VIBE_SCAN"
            self.status_msg = "✨ VIBE MODE: Reading Emotions..."
        else:
            self.state = "SCANNING"
            self.status_msg = "Scanning..."
            
        self.scan_start_time = time.time()
        self.collected_emotions = [] 
        self.final_stats = {} 
        profiler.reset()
        
        process = psutil.Process(os.getpid())
        self.start_mem = process.memory_info().rss / (1024 * 1024)
        
        self.cam_thread.start_scan()
        self.inf_thread.start_scan()

    def update_logic(self):
        if self.state == "SCANNING" or self.state == "VIBE_SCAN":
            results = self.inf_thread.get_results()
            if results:
                self.collected_emotions.extend(results)
                
                latest_label, latest_score = results[-1]
                if self.state == "SCANNING":
                    self.status_msg = f"Scanning... {latest_label} ({latest_score*100:.0f}%)"

            if time.time() - self.scan_start_time > self.scan_duration:
                self.finish_scan()

    def finish_scan(self):
        is_vibe = (self.state == "VIBE_SCAN")
        self.inf_thread.stop_scan()
        self.cam_thread.stop_scan()
        
        top_emotion = "None"
        avg_confidence = 0.0

        if self.collected_emotions:
            # Extract just names for counting the winner
            emotion_names = [e[0] for e in self.collected_emotions]
            counts = Counter(emotion_names)
            top_emotion = counts.most_common(1)[0][0]
            
            # Calculate avg confidence for the winner
            winner_scores = [e[1] for e in self.collected_emotions if e[0] == top_emotion]
            if winner_scores:
                avg_confidence = sum(winner_scores) / len(winner_scores)
        
        # Get Recommendations (Pass raw names, controller will filter for Winner only)
        emotion_names_only = [e[0] for e in self.collected_emotions]
        self.recommendations = self.music_ctrl.get_recommendations(emotion_names_only)
        
        avgs = profiler.get_averages()
        process = psutil.Process(os.getpid())
        end_mem = process.memory_info().rss / (1024 * 1024)
        
        self.final_stats = {
            "emotion": top_emotion,
            "confidence": avg_confidence, 
            "cam_ms": avgs['avg_cam_ms'], "pre_ms": avgs['avg_pre_ms'],
            "inf_ms": avgs['avg_inf_ms'], "db_ms":  avgs['total_db_ms'],
            "fps":    avgs['frames_processed'] / self.scan_duration,
            "mem_mb": end_mem - self.start_mem
        }
        self.log_benchmark_csv()

        if is_vibe:
            if self.recommendations:
                top_song_name, top_song_link = self.recommendations[0]
                self.music_ctrl.play_song(top_song_link)
                self.status_msg = f"✨ VIBE: Playing {top_song_name[:20]}..."
                self.state = "VIBE_PLAY"
            else:
                self.status_msg = "VIBE: No emotion found. Retrying..."
                self.start_scan(vibe_mode=True)
        else:
            self.state = "RESULTS"
            self.status_msg = f"Scan Complete. {top_emotion} ({avg_confidence*100:.0f}%)"

    def log_benchmark_csv(self):
        try:
            # LOG ONLY EMOTION LABEL (No Confidence Percentage in CSV)
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "detector": CONFIG['DETECTOR_TYPE'],
                "detected_emotion": self.final_stats.get('emotion', 'None'),
                "confidence_pct": f"{self.final_stats.get('confidence', 0)*100:.2f}%",
                "cam_input_ms": f"{self.final_stats.get('cam_ms', 0):.2f}",
                "preprocess_ms": f"{self.final_stats.get('pre_ms', 0):.2f}",
                "inference_ms": f"{self.final_stats.get('inf_ms', 0):.2f}",
                "fps_effective": f"{self.final_stats.get('fps', 0):.2f}"
            }
            # Target file: benchmark_log_v2.csv
            file_exists = os.path.isfile("benchmark_log_v2_webcam.csv")
            with open("benchmark_log_v2_webcam.csv", 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                if not file_exists: writer.writeheader()
                writer.writerow(log_entry)
            print(f"📄 Logged: {log_entry['detected_emotion']}")
        except Exception as e:
            print(f"⚠️ CSV Error: {e}")

    def draw(self):
        self.screen.fill((30, 30, 30))
        
        # --- 1. Draw Camera (Using LAYOUT Offset) ---
        frame = self.cam_thread.get_frame()
        if frame is not None:
            frame = cv2.resize(frame, CONFIG['CAM_SIZE'])
            color = (0, 255, 0)
            if "VIBE" in self.state: color = (255, 0, 255)
            if self.state in ["SCANNING", "VIBE_SCAN"]:
                cv2.rectangle(frame, (0,0), CONFIG['CAM_SIZE'], color, 10)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            surf = pygame.surfarray.make_surface(frame)
            self.screen.blit(surf, (LAYOUT['CAM_X'], LAYOUT['CAM_Y']))
        
        # --- 2. Status & Detailed Instructions (Bottom Section) ---
        status_y = LAYOUT['CAM_Y'] + CONFIG['CAM_SIZE'][1] + 20
        self.draw_text(f"Status: {self.status_msg}", (LAYOUT['CAM_X'], status_y), (0, 255, 255), size=35)

        # Instructions Bar at the bottom
        instr_y = CONFIG['SCREEN_H'] - 80
        if self.state == "IDLE":
            instr_text = "SPACE: Start Scan | V: Vibe Mode | S: Stop Music | Q: Quit"
        else:
            instr_text = "V: Toggle Vibe Mode | S: Stop Music | Q: Quit"
        self.draw_text(instr_text, (LAYOUT['CAM_X'], instr_y), (200, 200, 200), size=28)

        # --- 3. Latency Stats ---
        if self.final_stats:
            stats_y = status_y + 50
            bx1, bx2 = LAYOUT['CAM_X'], LAYOUT['CAM_X'] + 350
            c_head, c_val = (255, 255, 0), (200, 200, 200)

            self.draw_text("Performance Metrics:", (bx1, stats_y), c_head, 28)
            self.draw_text(f"Inference: {self.final_stats.get('inf_ms',0):.1f} ms", (bx1, stats_y + 30), c_val, 22)
            self.draw_text(f"FPS: {self.final_stats.get('fps',0):.1f}", (bx1, stats_y + 60), c_val, 22)
            
            emotion = self.final_stats.get('emotion', 'None')
            conf = self.final_stats.get('confidence', 0.0)
            self.draw_text(f"RESULT: {emotion.upper()} ({conf*100:.1f}%)", (bx2, stats_y + 30), (0, 255, 0), size=50)

        # --- 4. Sidebar Playlist ---
        sidebar_x = LAYOUT['SIDEBAR_X_OFFSET']
        self.draw_text("RECOMMENDATIONS", (sidebar_x, 50), (0, 255, 0), size=35)
        
        self.song_rects = []
        current_y = 110
        for i, (name, link) in enumerate(self.recommendations):
            if i > 22: break
            rect = pygame.Rect(sidebar_x, current_y, 500, LAYOUT['LINE_SPACING'])
            self.song_rects.append(rect)
            is_hover = rect.collidepoint(pygame.mouse.get_pos())
            color = (255, 255, 0) if is_hover else (255, 255, 255)
            display_name = (name[:35] + '...') if len(name) > 35 else name
            self.draw_text(f"{i+1}. {display_name}", (sidebar_x + 5, current_y + 5), color, size=24)
            current_y += LAYOUT['LINE_SPACING']

        pygame.display.flip()
    
    def draw_text(self, text, pos, color, size=24):
        font = pygame.font.Font(None, size)
        surf = font.render(text, True, color)
        self.screen.blit(surf, pos)

if __name__ == "__main__":
    app = App()
    app.run()
