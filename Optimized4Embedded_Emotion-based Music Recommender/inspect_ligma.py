import tensorflow as tf
tflite = tf.lite 
import numpy as np
import warnings # Import warnings module to handle the deprecation warning

# --- Configuration ---
TFLITE_MODEL_PATH = 'op_model.tflite' # Make sure this path is correct

# Suppress the specific LiteRT deprecation warning if desired
warnings.filterwarnings("ignore", message=".*LiteRT interpreter.*ai_edge_litert.*")

try:
    # --- Load the TFLite model and allocate tensors ---
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    print(f"✅ Model loaded: {TFLITE_MODEL_PATH}")

    # --- Get Input Tensor Details ---
    input_details = interpreter.get_input_details()
    print("\n--- Input Tensor Details ---")
    for detail in input_details:
        print(f"  Index: {detail['index']}")
        print(f"  Name: {detail.get('name', 'N/A')}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Data Type: {detail['dtype']}")
        
        # Corrected check for quantization parameters
        quant_params = detail.get('quantization_parameters', {})
        scales = quant_params.get('scales', np.array([])) # Default to empty array
        zero_points = quant_params.get('zero_points', np.array([])) # Default to empty array
        
        # Check if BOTH scale and zero_point arrays are NOT empty
        if scales.size > 0 and zero_points.size > 0:
            print(f"  Quantization: scale={scales}, zero_point={zero_points}")
        else:
            print("  Quantization: None (likely float32)")
        print("-" * 10)


    # --- Get Output Tensor Details ---
    output_details = interpreter.get_output_details()
    print("\n--- Output Tensor Details ---")
    for detail in output_details:
        print(f"  Index: {detail['index']}")
        print(f"  Name: {detail.get('name', 'N/A')}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Data Type: {detail['dtype']}")
        
        # Corrected check for quantization parameters
        quant_params = detail.get('quantization_parameters', {})
        scales = quant_params.get('scales', np.array([])) # Default to empty array
        zero_points = quant_params.get('zero_points', np.array([])) # Default to empty array

        # Check if BOTH scale and zero_point arrays are NOT empty
        if scales.size > 0 and zero_points.size > 0:
             print(f"  Quantization: scale={scales}, zero_point={zero_points}")
        else:
            print("  Quantization: None (likely float32)")
        print("-" * 10)

except Exception as e:
    print(f"❌ Failed to load or inspect model: {e}")