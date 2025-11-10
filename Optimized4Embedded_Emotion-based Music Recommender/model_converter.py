import tensorflow as tf
import numpy as np
import os
import traceback # For detailed error reporting

# --- Configuration ---
KERAS_MODEL_PATH = 'model.h5'
# New output file name to avoid overwriting your INT8 model
TFLITE_MODEL_PATH = 'op_model_float32.tflite' 

# --- Conversion Function (No Quantization) ---
def convert_model(keras_path, tflite_path):
    """
    Builds the Keras model architecture, loads weights, converts to a standard
    Float32 TFLite model, and saves it.
    """
    print(f"\n--- Starting Model Conversion (Float32) ---")
    print(f"Defining Keras model architecture...")

    try:        
        # 1. Define the model architecture (same as before)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25)) 

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5)) 

        model.add(tf.keras.layers.Dense(7, activation='softmax')) 

        print("✅ Model architecture defined.")
        model.summary()

        # 2. Load ONLY the weights
        print(f"Loading weights from: {keras_path}")
        if not os.path.exists(keras_path):
            print(f"❌ ERROR: Keras weights file not found at {keras_path}")
            return
        model.load_weights(keras_path)
        print("✅ Weights loaded successfully.")

        # 3. Initialize the TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # --- No optimizations applied, will default to Float32 ---
        print("Skipping quantization, converting to standard Float32.")

        # --- Convert the model ---
        print("\nConverting model to TFLite format...")
        tflite_model = converter.convert() # Standard conversion
        print("✅ Model converted successfully.")

        # --- Save the TFLite model ---
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ Float32 TFLite model saved to: {tflite_path}")

        # --- Compare file sizes ---
        try:
            keras_size = os.path.getsize(keras_path) / (1024*1024)
            tflite_size = os.path.getsize(tflite_path) / (1024*1024)
            print(f"Original Keras weights size: {keras_size:.2f} MB")
            print(f"Float32 TFLite model size: {tflite_size:.2f} MB")
        except Exception as size_e:
            print(f"Could not calculate file sizes: {size_e}")

        print("\n--- Model Conversion Complete ---")

    except Exception as e:
        print(f"❌ An error occurred during conversion: {e}")
        traceback.print_exc()

# --- Main execution ---
if __name__ == "__main__":
    convert_model(KERAS_MODEL_PATH, TFLITE_MODEL_PATH)