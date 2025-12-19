import cv2

print("Attempting to open camera...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("? Failed to open camera!")
else:
    print("? Camera opened successfully.")
    
    ret, frame = cap.read()
    if ret:
        print(f"? Captured frame of size: {frame.shape}")
        cv2.imwrite("opencv_test.jpg", frame)
        print("Saved opencv_test.jpg")
    else:
        print("? Camera opened, but failed to capture frame (ret=False).")

cap.release()