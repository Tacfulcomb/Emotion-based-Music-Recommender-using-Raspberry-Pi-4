# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
import tensorflow as tf
import glob

from collections import Counter
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers  import Conv2D
from tensorflow.python.keras.layers  import MaxPooling2D
import base64
import tempfile
import os


## === GPU / CPU selection flag ===
USE_GPU = False   # 👈 set True to use GPU, False to force CPU

if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU
    print("Running on CPU only.")
else:
    print("Attempting to use GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU detected: {gpus[0].name}")
        except RuntimeError as e:
            print("⚠️ GPU setup failed, fallback to CPU:", e)
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        print("❌ No GPU detected, using CPU.")

df = pd.read_csv("Emotion-based-music-recommendation-system/muse_v3.csv")

df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df = df[['name','emotional','pleasant','link','artist']]
# print(df)

df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index()
# print(df)

df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

def fun(list):

    data = pd.DataFrame()

    if len(list) == 1:
        v = list[0]
        t = 30
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v == 'Angry':
             data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v == 'fear':
            data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif v == 'happy':
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)

    elif len(list) == 2:
        times = [30,20]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':    
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':              
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':             
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])

    elif len(list) == 3:
        times = [55,20,15]
        for i in range(len(list)): 
            v = list[i]          
            t = times[i]

            if v == 'Neutral':              
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':               
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':             
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':               
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:      
                data = pd.concat([df_sad.sample(n=t)])


    elif len(list) == 4:
        times = [30,29,18,9]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral': 
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':              
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':              
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':               
                data =pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])
    else:
        times = [10, 7, 6, 5, 2]
        for i in range(len(list)):
            v = list[i]
            # use times[i] when available, otherwise fallback to the last value
            t = times[i] if i < len(times) else times[-1]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([data, df_sad.sample(n=t)], ignore_index=True)

    # print("data of list func... :",data)
    return data

def pre(l):

    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    # print("Processed Emotions:", result)

    # result = [item for items, c in Counter(l).most_common()
    #           for item in [items] * c]

    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
            print(result)
    # print("Return the list of unique emotions in the order of occurrence frequency :",ul)
    return ul
    




model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))


model.load_weights('Emotion-based-music-recommendation-system/model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# cv2.ocl.setUseOpenCL(False)

print("Loading Haarcascade Classifier...")
face = cv2.CascadeClassifier('Emotion-based-music-recommendation-system/haarcascade_frontalface_default.xml')
if face.empty():
    print("Haarcascade Classifier failed to load.")
else:
    print("Haarcascade Classifier loaded successfully.")

page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion based music recommendation</b></h2>"
            , unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>"
            , unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)

list = []
with col1:
    pass
with col2:
    if 'scanning' not in st.session_state:
        st.session_state.scanning = False

    scan_btn = st.button('SCAN EMOTION (Click here)')
    if scan_btn and not st.session_state.scanning:
        st.session_state.scanning = True
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("No camera available. Connect a camera or upload a video and try again.")
            st.session_state.scanning = False
        else:
            list.clear()
            placeholder = st.empty()
            max_frames = 20
            frame_count = 0

            # === TESTBENCH START ===
            import psutil, time
            process = psutil.Process(os.getpid())
            start_mem = process.memory_info().rss / (1024*1024)
            start_time = time.time()
            total_inference_time = 0
            num_inferences = 0
            # === TESTBENCH START ===

            with st.spinner("Scanning emotions..."):
                while frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                    for (x, y, w, h) in faces:
                        roi_gray = gray[y:y + h, x:x + w]
                        try:
                            cropped_img = cv2.resize(roi_gray, (48, 48))
                            cropped_img = np.expand_dims(cropped_img, -1)
                            cropped_img = np.expand_dims(cropped_img, 0).astype('float32') / 255.0

                            # --- Time inference ---
                            t0 = time.time()
                            prediction = model.predict_on_batch(cropped_img)
                            t1 = time.time()
                            total_inference_time += (t1 - t0)
                            num_inferences += 1
                            # --- Time inference ---

                        except Exception as e:
                            print("Prediction failed:", e)
                            prediction = np.zeros((1, 7))

                        max_index = int(np.argmax(prediction))
                        list.append(emotion_dict[max_index])
                        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                        cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    small = cv2.resize(rgb, (640, 360))
                    placeholder.image(small, channels="RGB")
                    frame_count += 1

            cap.release()
            st.session_state.scanning = False

            # === TESTBENCH END ===
            end_time = time.time()
            end_mem = process.memory_info().rss / (1024*1024)
            total_time = end_time - start_time
            avg_inference = (total_inference_time / num_inferences) * 1000 if num_inferences else 0
            fps = num_inferences / total_time if total_time > 0 else 0
            mem_used = end_mem - start_mem

            st.markdown("---")
            st.markdown(f"### ⚙️ Performance Summary")
            st.write(f"**Average Inference Time:** {avg_inference:.2f} ms/frame")
            st.write(f"**Approx FPS:** {fps:.2f}")
            st.write(f"**Memory Used:** {mem_used:.2f} MB")
            st.write(f"**Frames Processed:** {num_inferences}")
            st.markdown("---")
            # === Log benchmark results ===
            try:
                import pandas as pd
                from datetime import datetime

                log_entry = pd.DataFrame([{
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "avg_inference_ms": avg_inference,
                    "fps": fps,
                    "mem_used_MB": mem_used,
                    "frames": num_inferences
                }])

                file_exists = os.path.isfile("benchmark_log.csv")
                log_entry.to_csv("benchmark_log.csv", mode='a', header=not file_exists, index=False)
                st.success("✅ Benchmark data recorded to benchmark_log.csv")

            except Exception as e:
                st.warning(f"⚠️ Could not save benchmark log: {e}")
            # === TESTBENCH END ===

with col3:
    pass

new_df = fun(list)
st.write("")

st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended song's with artist names</b></h5>"
            , unsafe_allow_html=True)

st.write("---------------------------------------------------------------------------------------------------------------------")

try:
  
    for l,a,n,i in zip(new_df["link"],new_df['artist'],new_df['name'],range(30)):

        st.markdown("""<h4 style='text-align: center;'><a href={}>{} - {}</a></h4>"""
                    .format(l,i+1,n),unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center; color: grey;'><i>{}</i></h5>" 
                    .format(a), unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------------------------------------")
except:
    pass