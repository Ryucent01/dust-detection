import cv2
import time
import sys
import torch
import threading
import queue
import os
import shutil
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# Try importing supabase
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    print("[WARN] 'supabase' library not found. Offline mode only.")
    SUPABASE_AVAILABLE = False

# ==========================================
# CONFIGURATION
# ==========================================
# IMPORTANT: Ensure your model file is in the same directory or provide full path
MODEL_PATH = "best_dust.pt"       
CONFIDENCE_THRESHOLD = 0.03       # Confidence threshold for detection
IMAGE_SIZE = 1280                  # Inference image size (YOLO default usually 640)
USE_CSI_CAMERA = False            # Set to True for Jetson CSI Camera (e.g. Raspberry Pi Cam), False for USB Webcam
CAMERA_INDEX = 0                  # USB Camera index (usually 0 or 1)
SENSOR_ID = 0                     # CSI Camera sensor ID (usually 0)
FLIP_METHOD = 0                   # 0=none, 2=180 rotation (adjust if camera is upside down)

# --- OFFLINE SYNC CONFIG ---
SUPABASE_URL = "https://mtowlknszgcwszbupenq.supabase.co"
SUPABASE_KEY = "sb_publishable_ChO3Id_bgOdYmJUHBqVy2Q_IHhst_DN"
BUCKET_NAME = "dust-images"
LOCAL_STORAGE_DIR = "data_storage"
# ==========================================

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    """
    Returns a GStreamer pipeline string for capturing from the CSI camera on Jetson devices.
    Adjust parameters based on your specific camera hardware (e.g., IMX219, IMX477).
    """
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class ImageSyncSystem:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        
        # Setup directories
        self.base_dir = Path(LOCAL_STORAGE_DIR)
        self.pending_dir = self.base_dir / "pending"
        self.uploaded_dir = self.base_dir / "uploaded"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.uploaded_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Supabase
        self.supabase = None
        if SUPABASE_AVAILABLE and "YOUR_SUPABASE_URL" not in SUPABASE_URL:
             try:
                 self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                 print("[INFO] Connected to Supabase.")
             except Exception as e:
                 print(f"[ERROR] Supabase Connection Failed: {e}")

        # Start Worker Thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        print("[INFO] Sync System Started (Background Thread).")

    def save_snapshot(self, frame):
        """
        Non-blocking: Puts frame into queue and returns immediately.
        """
        if frame is None: return
        self.queue.put(frame.copy())
        print("[INFO] Image queued for background saving...")

    def _process_queue(self):
        """
        Worker loop: Saves to disk -> Uploads -> Moves file.
        """
        while self.running:
            try:
                # Wait for a frame (blocking)
                frame = self.queue.get()
                
                # 1. Generate Filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"dust_{timestamp}.jpg"
                pending_path = self.pending_dir / filename
                
                # 2. Save to Disk (Pending)
                cv2.imwrite(str(pending_path), frame)
                # print(f"[DEBUG] Saved local: {filename}")
                
                # 3. Attempt Upload (if configured)
                uploaded = False
                if self.supabase:
                    try:
                        with open(pending_path, 'rb') as f:
                            self.supabase.storage.from_(BUCKET_NAME).upload(
                                path=filename,
                                file=f,
                                file_options={"content-type": "image/jpeg"}
                            )
                        uploaded = True
                        print(f"[SUCCESS] Uploaded: {filename}")
                    except Exception as e:
                        print(f"[WARN] Upload failed for {filename}: {e}")
                
                # 4. Move if uploaded
                if uploaded:
                    shutil.move(str(pending_path), str(self.uploaded_dir / filename))
                
                self.queue.task_done()
                
            except Exception as e:
                print(f"[ERROR] Sync Worker Error: {e}")

class DustDetector:
    def __init__(self):
        """
        Initialize the Dust Detector application:
        1. Check CUDA availability for GPU acceleration.
        2. Load the YOLOv8 model.
        3. Setup the video source (USB or CSI).
        """
        print("====================================")
        print("   Initializing Dust Detector...")
        print("====================================")
        
        # 1. Device Setup - Auto-detect CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {self.device}")
        if self.device == 'cuda':
            print(f"[INFO] CUDA Device: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARNING] CUDA not detected. Running on CPU (will be slow).")

        # 2. Load Model
        try:
            print(f"[INFO] Loading model from '{MODEL_PATH}'...")
            self.model = YOLO(MODEL_PATH)
            
            # Optional: warmup to reduce delay on first inference
            print("[INFO] Warming up model...")
            dummy_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(self.device)
            self.model.predict(source=dummy_input, verbose=False) 
            print("[INFO] Model loaded successfully.")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print(f"Please ensure '{MODEL_PATH}' exists or update MODEL_PATH in the script.")
            sys.exit(1)

        # 3. Camera Setup
        self.cap = None
        self.init_camera()

        # Application State
        self.is_live = True          # True = Live Preview, False = Frozen/Inference Mode
        self.last_frame = None       # Stores the last captured frame
        self.processed_img = None    # Stores the result of inference
        
        # 4. Initialize Sync System
        self.sync_system = ImageSyncSystem()

    def init_camera(self):
        """
        Initialize the video capture object based on configuration.
        """
        if USE_CSI_CAMERA:
            print("[INFO] Attempting to open CSI Camera via GStreamer...")
            pipeline = gstreamer_pipeline(
                sensor_id=SENSOR_ID,
                flip_method=FLIP_METHOD
            )
            print(f"[DEBUG] Pipeline: {pipeline}")
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            print(f"[INFO] Attempting to open USB Camera (Index {CAMERA_INDEX})...")
            self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():
            print("[ERROR] Could not open video device.")
            sys.exit(1)
            
        # Basic camera properties
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[INFO] Camera opened: {int(width)}x{int(height)}")

    def add_watermark(self, image):
        """
        Adds a transparent 'Ryucent Tech' watermark to the bottom-right corner.
        """
        if image is None: return image
        
        # Text Setup
        text = "Ryucent Tech"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255) # White
        
        # Calculate Position (Bottom Right)
        h, w = image.shape[:2]
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        x = w - text_w - 20
        y = h - 20
        
        # Create Overlay for Transparency
        overlay = image.copy()
        cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness)
        
        # Blend
        alpha = 0.5  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        return image

    def perform_detection(self, frame):
        """
        Runs YOLO inference on the given frame.
        Returns the annotated image with bounding boxes.
        """
        print("-" * 30)
        print("[INFO] Running Inference...")
        
        start_time = time.time()
        
        # Run prediction
        # stream=False ensures we get all results at once
        results = self.model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=self.device,
            verbose=False
        )
        
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        
        # Process results
        result = results[0]
        num_detections = len(result.boxes)
        
        print(f"[RESULT] Inference Time: {inference_time_ms:.1f} ms")
        print(f"[RESULT] Detected Objects: {num_detections}")
        
        # Generate the annotated frame (draws boxes/labels)
        annotated_frame = result.plot()
        
        # Add overlay text for performance stats
        cv2.putText(annotated_frame, f"Objects: {num_detections}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Time: {inference_time_ms:.1f} ms", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return annotated_frame

    def start_loop(self):
        """
        Main application loop.
        """
        print("\n========================================")
        print("          APP STARTED")
        print("========================================")
        print(" Controls:")
        print("  [SPACE]    : Toggle between Live View and Detection (Freezes frame)")
        print("  [q] or [ESC] : Exit application")
        print("========================================\n")

        try:
            while True:
                # -------------------
                # LIVE PREVIEW MODE
                # -------------------
                if self.is_live:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("[ERROR] Failed to read frame from camera.")
                        time.sleep(1)
                        continue
                    
                    self.last_frame = frame.copy() # Save copy for potential processing
                    display_image = frame
                    
                    # Overlay "LIVE" indicator
                    cv2.putText(display_image, "LIVE VIEW - Press SPACE to Detect", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # Orange color
                
                # -------------------
                # FROZEN / INFERENCE MODE
                # -------------------
                else:
                    # If we just switched to frozen and haven't processed yet
                    if self.processed_img is None:
                        if self.last_frame is not None:
                            self.processed_img = self.perform_detection(self.last_frame)
                        else:
                            print("[WARN] No frame to process. Returning to live mode.")
                            self.is_live = True
                            continue
                    
                    display_image = self.processed_img
                    
                    # Overlay instruction to return
                    h, w = display_image.shape[:2]
                    cv2.putText(display_image, "FROZEN RESULT - Press SPACE to Resume", (20, h - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Show the frame
                if display_image is not None:
                    # Apply watermark to the DISPLAY ONLY (not the saved data)
                    self.add_watermark(display_image)
                    cv2.imshow("Dust Detection - Jetson Orin Nano", display_image)

                # -------------------
                # KEY HANDLING
                # -------------------
                key = cv2.waitKey(1) & 0xFF
                
                # SPACEBAR (32) to toggle modes
                if key == 32: 
                    self.is_live = not self.is_live
                    
                    if not self.is_live:
                        # Transitioning Live -> Frozen
                        # 1. Trigger Background Save (Non-blocking)
                        if self.last_frame is not None:
                            self.sync_system.save_snapshot(self.last_frame)
                            
                        # 2. Reset processed_img so it triggers a fresh detection
                        self.processed_img = None 
                    else:
                        # Transitioning Frozen -> Live
                        print("[INFO] Resuming Live View...")
                
                # 'q' (113) or ESC (27) to quit
                elif key == ord('q') or key == 27:
                    print("[INFO] Exit signal received.")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Keyboard Interrupt received.")
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Release camera resources and close windows.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete. Goodbye!")

if __name__ == "__main__":
    app = DustDetector()
    app.start_loop()
