import cv2
import numpy as np
# Import your modified object detection function 
from backend.agents.vision import detect_objects 

def run_live_monitor():
    # For an external IP camera, replace 0 with: "rtsp://username:password@IP:PORT/h264"
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not connect to the video source.")
        return

    print("[INFO] Camera stream started. Press 'q' on the video window to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Dropped frame encountered. Reconnecting...")
            continue

        # 1. Compress the raw OpenCV frame (numpy array) into standard JPEG bytes
        success, encoded_image = cv2.imencode(".jpg", frame)
        if not success:
            continue
        image_bytes = encoded_image.tobytes()

        # 2. Process frame using your Hugging Face framework
        result = detect_objects(image_bytes)

        if result and "image" in result:
            # 3. Decode the returned annotated bytes back to a numpy array for visualization
            np_img = np.frombuffer(result["image"], np.uint8)
            annotated_frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            # 4. Show metrics natively on the video window frame title
            window_title = f"Live Monitor | Objects: {len(result['objects'])} | Latency: {result['latency']}s"
            cv2.imshow(window_title, annotated_frame)

        # Handle keyboard break sequence (Press 'q' to gracefully turn off the camera)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera interface terminated smoothly.")

if __name__ == "__main__":
    run_live_monitor()