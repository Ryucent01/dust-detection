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
import numpy as np

# Try importing Firebase
try:
    import firebase_admin
    from firebase_admin import credentials, storage, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    print("[WARN] 'firebase-admin' library not found. Offline mode only.")
    FIREBASE_AVAILABLE = False

# ==========================================
# CONFIGURATION
# ==========================================
# IMPORTANT: Ensure your model file is in the same directory or provide full path
MODEL_PATH = "best_dust.pt"       
CONFIDENCE_THRESHOLD = 0.03       # Confidence threshold for detection
IMAGE_SIZE = 1280                  # Inference image size (YOLO default usually 640)
USE_BASLER_CAMERA = True          # Set to True for Basler GigE Camera
USE_CSI_CAMERA = False            # Set to True for Jetson CSI Camera (e.g. Raspberry Pi Cam), False for USB Webcam
CAMERA_INDEX = 0                  # USB Camera index (usually 0 or 1)
SENSOR_ID = 0                     # CSI Camera sensor ID (usually 0)
FLIP_METHOD = 0                   # 0=none, 2=180 rotation (adjust if camera is upside down)

# --- OFFLINE SYNC CONFIG ---
# --- FIREBASE CONFIG ---
# Place your 'serviceAccountKey.json' in the project folder
SERVICE_ACCOUNT_KEY = "serviceAccountKey.json"
FIREBASE_STORAGE_BUCKET = "ryucent-rg-dust.firebasestorage.app"
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
        
        # Setup Firebase
        self.db = None
        self.bucket = None
        if FIREBASE_AVAILABLE and os.path.exists(SERVICE_ACCOUNT_KEY):
             try:
                 cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
                 firebase_admin.initialize_app(cred, {
                     'storageBucket': FIREBASE_STORAGE_BUCKET
                 })
                 self.db = firestore.client()
                 self.bucket = storage.bucket()
                 print("[INFO] Connected to Firebase.")
             except Exception as e:
                 print(f"[ERROR] Firebase Connection Failed: {e}")
        else:
            if not os.path.exists(SERVICE_ACCOUNT_KEY):
                print(f"[WARN] '{SERVICE_ACCOUNT_KEY}' missing. Firebase disabled.")

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
                
                # 3. Attempt Upload to Firebase
                uploaded = False
                if self.db and self.bucket:
                    try:
                        # Upload Image to Storage
                        blob = self.bucket.blob(f"raw_images/{filename}")
                        blob.upload_from_filename(str(pending_path))
                        
                        # Create Record in Firestore
                        self.db.collection("images").document(filename).set({
                            "filename": filename,
                            "timestamp": firestore.SERVER_TIMESTAMP,
                            "status": "pending",
                            "storage_path": f"raw_images/{filename}"
                        })
                        
                        uploaded = True
                        print(f"[SUCCESS] Firebase Upload: {filename}")
                    except Exception as e:
                        print(f"[WARN] Firebase upload failed for {filename}: {e}")
                
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

        # 5. Load Company Logo
        self.logo = None
        self.logo_path = "logo.png"
        if os.path.exists(self.logo_path):
            try:
                # Load with alpha channel
                logo_raw = cv2.imread(self.logo_path, cv2.IMREAD_UNCHANGED)
                if logo_raw is not None:
                    # PRO STEP: Auto-crop transparent margins to ensure it hits the corner perfectly
                    if logo_raw.shape[2] == 4:
                        alpha = logo_raw[:, :, 3]
                        coords = cv2.findNonZero(alpha)
                        if coords is not None:
                            x, y, w, h = cv2.boundingRect(coords)
                            logo_raw = logo_raw[y:y+h, x:x+w]
                    
                    # Store original for splash screen, and resized for overlay
                    self.logo_original = logo_raw.copy()
                    
                    target_width = 180
                    h, w = logo_raw.shape[:2]
                    target_height = int(h * (target_width / w))
                    self.logo = cv2.resize(logo_raw, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                    print(f"[INFO] Logo loaded and optimized.")
            except Exception as e:
                print(f"[WARN] Failed to load logo: {e}")

        # 6. Setup Window
        self.window_name = "Dust Detection - Jetson Orin Nano"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL) 

        # 7. Cinematic Splash Screen
        self.show_splash_screen()

    def show_splash_screen(self):
        """
        Displays an elegant, smooth logo fade-in and fade-out.
        """
        if not hasattr(self, 'logo_original'): return
        
        canvas_h, canvas_w = 720, 1280
        # High quality logo sizing for splash
        s_width = 450
        sh, sw = self.logo_original.shape[:2]
        s_height = int(sh * (s_width / sw))
        splash_logo = cv2.resize(self.logo_original, (s_width, s_height), interpolation=cv2.INTER_LANCZOS4)
        
        lx = (canvas_w - s_width) // 2
        ly = (canvas_h - s_height) // 2

        # Animation: Fade In (60 frames) -> Hold (20 frames) -> Fade Out (40 frames)
        phases = [
            (range(60), lambda i: i / 60.0),        # Fade In
            (range(20), lambda i: 1.0),             # Hold
            (range(40), lambda i: 1.0 - (i / 40.0)) # Fade Out
        ]

        for frames, alpha_func in phases:
            for i in frames:
                frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                alpha = alpha_func(i)
                
                # Apply Logo with Alpha
                roi = frame[ly:ly+s_height, lx:lx+s_width]
                l_bgr = splash_logo[:, :, :3]
                l_mask = (splash_logo[:, :, 3].astype(float) / 255.0) * alpha
                
                for c in range(3):
                    roi[:, :, c] = (l_bgr[:, :, c] * l_mask).astype(np.uint8)
                
                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(12) & 0xFF == ord('q'): return
            
        time.sleep(0.2)

    def init_camera(self):
        """
        Initialize the video capture object based on configuration.
        """
        self.use_basler = USE_BASLER_CAMERA
        
        if self.use_basler:
            print("[INFO] Attempting to open Basler GigE Camera...")
            try:
                from pypylon import pylon
                self.tl_factory = pylon.TlFactory.GetInstance()
                devices = self.tl_factory.EnumerateDevices()
                if len(devices) == 0:
                    print("[ERROR] No Basler cameras found on the network.")
                    sys.exit(1)
                
                self.camera = pylon.InstantCamera(self.tl_factory.CreateFirstDevice())
                self.camera.Open()
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
                
                self.converter = pylon.ImageFormatConverter()
                # Check if camera is monochrome or color to set appropriate conversion
                pixel_format = self.camera.PixelFormat.GetValue()
                if "mono" in self.camera.GetDeviceInfo().GetModelName().lower() or \
                   pixel_format.startswith("Mono"):
                    self.converter.OutputPixelFormat = pylon.PixelType_Mono8
                else:
                    self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                
                self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
                
                print(f"[INFO] Basler Camera opened: {self.camera.GetDeviceInfo().GetModelName()}")
                return
            except ImportError:
                print("[ERROR] pypylon is not installed. Run 'pip install pypylon'.")
                sys.exit(1)
            except Exception as e:
                print(f"[ERROR] Could not open Basler camera: {e}")
                sys.exit(1)

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

    def add_logo(self, image):
        """
        Premium overlay: Adds the logo to the bottom-right corner with a sharp, clear drop shadow.
        """
        if image is None or self.logo is None:
            return image
        
        img_h, img_w = image.shape[:2]
        logo_h, logo_w = self.logo.shape[:2]
        
        # Position: Elevated from corner for a modern look (15px padding)
        padding = 15
        x_offset = img_w - logo_w - padding
        y_offset = img_h - logo_h - padding
        
        if x_offset < 0 or y_offset < 0: return image

        # Get ROI
        roi = image[y_offset:y_offset + logo_h, x_offset:x_offset + logo_w]
        
        if self.logo.shape[2] == 4:
            logo_bgr = self.logo[:, :, :3]
            logo_mask = self.logo[:, :, 3]
            
            # --- High Visibility Shadow ---
            shadow_mask = cv2.GaussianBlur(logo_mask, (5, 5), 0)
            sx, sy = x_offset + 2, y_offset + 2
            
            if sx + logo_w < img_w and sy + logo_h < img_h:
                shadow_roi = image[sy:sy + logo_h, sx:sx + logo_w]
                mask_norm = shadow_mask.astype(float) / 255.0
                for c in range(3):
                    shadow_roi[:, :, c] = (shadow_roi[:, :, c] * (1.0 - mask_norm * 0.6)).astype(image.dtype)
            
            # --- Sharp Logo Overlay ---
            mask_norm = logo_mask.astype(float) / 255.0
            for c in range(3):
                roi[:, :, c] = (roi[:, :, c] * (1.0 - mask_norm) + logo_bgr[:, :, c] * mask_norm).astype(image.dtype)
        else:
            cv2.addWeighted(self.logo, 0.9, roi, 0.1, 0, roi)
            
        return image

    def draw_modern_ui(self, image, detections=None, inference_time=None):
        """
        Renders minimalist, floating UI elements without bulky backgrounds.
        """
        if image is None: return image
        
        # 1. Floating Status Indicator
        pulse_color = (0, 0, 255) if self.is_live else (0, 255, 255)
        status_text = "LIVE" if self.is_live else "ANALYSIS"
        
        # Draw text with a subtle shadow for visibility on any background
        def draw_text_with_shadow(img, text, pos, scale, color, thick):
            # Shadow (black)
            cv2.putText(img, text, (pos[0]+1, pos[1]+1), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
            # Main Text
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)

        # Pulse dot
        cv2.circle(image, (25, 30), 5, pulse_color, -1)
        draw_text_with_shadow(image, status_text, (40, 36), 0.6, (255, 255, 255), 1)
        
        # 2. Minimalist Stats (Stacked vertically)
        if detections is not None:
            draw_text_with_shadow(image, f"OBJECTS: {detections}", (25, 65), 0.5, (0, 255, 0), 1)
        
        if inference_time is not None:
            draw_text_with_shadow(image, f"{inference_time:.1f}ms", (25, 85), 0.4, (200, 200, 200), 1)
            
        return image

    def perform_detection(self, frame):
        """
        Runs YOLO inference on the given frame.
        Returns the annotated image PLUS metadata.
        """
        print("-" * 30)
        print("[INFO] Running Inference...")
        
        start_time = time.time()
        
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
        
        # Generate the annotated frame
        annotated_frame = result.plot()
        
        # Return frame and metadata for modern UI rendering
        return annotated_frame, num_detections, inference_time_ms

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

        self.last_det_count = None
        self.last_det_time = None

        try:
            while True:
                # -------------------
                # LIVE PREVIEW MODE
                # -------------------
                if self.is_live:
                    if hasattr(self, 'use_basler') and self.use_basler:
                        try:
                            from pypylon import pylon
                            if self.camera.IsGrabbing():
                                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                                if grabResult.GrabSucceeded():
                                    image = self.converter.Convert(grabResult)
                                    frame = image.GetArray()
                                    
                                    # If monochrome, convert to 3-channel BGR for YOLO compatibility
                                    if len(frame.shape) == 2:
                                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                                        
                                    grabResult.Release()
                                else:
                                    print(f"[ERROR] Grab failed: {grabResult.ErrorCode} {grabResult.ErrorDescription}")
                                    grabResult.Release()
                                    continue
                            else:
                                print("[ERROR] Basler camera is not grabbing.")
                                time.sleep(1)
                                continue
                        except Exception as e:
                            print(f"[ERROR] Basler capture error: {e}")
                            time.sleep(1)
                            continue
                    else:
                        ret, frame = self.cap.read()
                        if not ret:
                            print("[ERROR] Failed to read frame from camera.")
                            time.sleep(1)
                            continue
                    
                    self.last_frame = frame.copy()
                    display_image = frame
                    
                    # Clean UI: No bulky instruction text here anymore
                
                # -------------------
                # FROZEN / INFERENCE MODE
                # -------------------
                else:
                    if self.processed_img is None:
                        if self.last_frame is not None:
                            self.processed_img, self.last_det_count, self.last_det_time = self.perform_detection(self.last_frame)
                        else:
                            print("[WARN] No frame to process. Returning to live mode.")
                            self.is_live = True
                            continue
                    
                    display_image = self.processed_img

                # Show the frame
                if display_image is not None:
                    # IMPORTANT: Create a COPY for display to avoid double-watermarking the same frame
                    render_frame = display_image.copy()
                    
                    # Apply Modern Interface
                    self.draw_modern_ui(render_frame, 
                                       detections=self.last_det_count if not self.is_live else None,
                                       inference_time=self.last_det_time if not self.is_live else None)
                    
                    # Apply logo (elevated and high quality)
                    self.add_logo(render_frame)
                    cv2.imshow(self.window_name, render_frame)

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
        if hasattr(self, 'use_basler') and self.use_basler:
            if hasattr(self, 'camera') and self.camera.IsOpen():
                self.camera.StopGrabbing()
                self.camera.Close()
        else:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete. Goodbye!")

if __name__ == "__main__":
    app = DustDetector()
    app.start_loop()
