

import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import importlib
import torch

# ============================================================
# CONFIG — ENABLE PLUGINS
# ============================================================

ENABLED_PLUGINS = ["waving","raise"]   # <-- Add your plugin names here

MODEL_PATH = r"D:\child detections\Child-posture\Project\posture_framework\models\yolov8s-pose.pt"
CAMERA_ID = 0

TARGET_WIDTH = 640
TARGET_HEIGHT = 480

print("\n" + "="*50)
print("        Aman's FINAL Posture Detection System")
print("="*50 + "\n")

# ============================================================
# GPU CHECK
# ============================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"[INFO] GPU Detected → {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("[INFO] Running on CPU")

print(f"[INFO] Device set to: {DEVICE.upper()}\n")

# ============================================================
# LOAD YOLO MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("[ERROR] YOLO model not found!")
    print(f"[ERROR] Path: {MODEL_PATH}")
    sys.exit(1)

print("[INFO] Loading YOLO pose model...")
try:
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    
    if model.model.yaml.get("kpt_shape", None) is None:
        print("[ERROR] This is NOT a pose model. Please use yolov8s-pose.pt")
        sys.exit(1)
        
    print("[SUCCESS] YOLO pose model loaded and verified")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model: {e}")
    sys.exit(1)

# ============================================================
# MEDIAPIPE HANDS INITIALIZATION                             |
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("[INFO] MediaPipe hands initialized\n")

# ============================================================
# LOAD PLUGINS                                               |
# ============================================================

PLUGIN_FOLDER = "posture_framework.plugins"
plugins = []
loaded_plugin_names = []

print("-" * 40)
print("[INFO] Loading Plugins:")
print("-" * 40)

if not ENABLED_PLUGINS:
    print("[WARNING] No plugins specified in ENABLED_PLUGINS")
else:
    for plugin_name in ENABLED_PLUGINS:
        try:
            module = importlib.import_module(f"{PLUGIN_FOLDER}.{plugin_name}")
            if hasattr(module, "Plugin"):
                plugin_instance = module.Plugin()
                plugins.append(plugin_instance)
                
                # Get plugin name for display
                plugin_display_name = getattr(plugin_instance, 'name', plugin_name.capitalize())
                loaded_plugin_names.append(plugin_display_name)
                print(f"  ✓ {plugin_display_name}")
            else:
                print(f"  ✗ {plugin_name}: Plugin class not found")
        except ModuleNotFoundError:
            print(f"  ✗ {plugin_name}: Module not found (check plugin folder)")
        except Exception as e:
            print(f"  ✗ {plugin_name}: {e}")

print("-" * 40)

if plugins:
    print(f"[SUCCESS] {len(plugins)} plugin(s) loaded successfully")
else:
    print("[WARNING] No plugins loaded successfully")

print("\n" + "="*50)
print("Starting Detection System...")
print("="*50 + "\n")

# ============================================================
# CAMERA INITIALIZATION                                      |
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("[ERROR] Cannot open camera!")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

print("[INFO] Camera started (Press 'Q' to quit)")
print("[INFO] Resolution:", TARGET_WIDTH, "x", TARGET_HEIGHT)
print()

# ============================================================
# RUN PLUGINS FUNCTION                                       |
# ============================================================

def run_plugins(frame, annotated, kpts, hand_res):
    """Run all loaded plugins and collect their messages"""
    messages, colors = [], []

    for plugin in plugins:
        try:
            msg, color = plugin.run(frame, annotated, kpts, hand_res)
            
            if msg and color and isinstance(color, (tuple, list)) and len(color) == 3:
                messages.append(str(msg))
                colors.append(tuple(int(c) for c in color))
        except Exception as e:
            print(f"[PLUGIN ERROR] {plugin.__class__.__name__}: {e}")
    
    return messages, colors

# ============================================================
# MAIN LOOP                                                  |
# ============================================================

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame")
        break
    
    frame_count += 1
    
    # Resize frame for processing
    frame_small = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    
    # ====================================================
    # YOLO POSE DETECTION                                |
    # ====================================================
    try:
        results = model(frame_small, verbose=False, device=DEVICE)
        annotated = results[0].plot()
        
        # Extract keypoints
        try:
            arr = results[0].keypoints.data.cpu().numpy()
            kpts = arr[0] if len(arr) else None
        except:
            kpts = None
    except Exception as e:
        print(f"[YOLO ERROR] {e}")
        annotated = frame_small.copy()
        kpts = None
    
    # ====================================================
    # HAND DETECTION (MediaPipe based)                   |
    # ====================================================
    rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    hand_res = hands.process(rgb)
    
    # Draw hand landmarks
    if hand_res and hand_res.multi_hand_landmarks:
        for hand_landmarks in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                annotated, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
            )
    
    # ====================================================
    # RUN PLUGINS                                        |
    # ====================================================
    plugin_msgs, plugin_colors = run_plugins(frame_small, annotated, kpts, hand_res)
    
    # ====================================================
    # DISPLAY SECTION                                    |
    # ====================================================
    
    # Top status bar background
    cv2.rectangle(annotated, (0, 0), (TARGET_WIDTH, 60), (30, 30, 30), -1)
    cv2.rectangle(annotated, (0, 0), (TARGET_WIDTH, 60), (60, 60, 60), 2)
    
    # ----------------------------------------------------
    # LEFT SIDE: SYSTEM TITLE AND STATUS         
    # ----------------------------------------------------
    
    # System title
    cv2.putText(annotated, "Posture Detection", (10, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    # Device info below title
    device_color = (0, 255, 0) if DEVICE == "cuda" else (255, 165, 0)
    cv2.putText(annotated, f"({DEVICE.upper()})", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, device_color, 1)
    
    # ----------------------------------------------------
    # CENTER: PLUGINS ENABLED (MAIN DISPLAY)
    # ----------------------------------------------------
    
    if loaded_plugin_names:
        # Show "Plugins Enabled:" in center top
        plugins_text = "Plugins Enabled: " + ", ".join(loaded_plugin_names)
        
        # Calculate text position for center
        text_size = cv2.getTextSize(plugins_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (TARGET_WIDTH - text_size[0]) // 2
        
        # Draw background for better visibility
        bg_padding = 10
        cv2.rectangle(annotated, 
                     (text_x - bg_padding, 15 - bg_padding),
                     (text_x + text_size[0] + bg_padding, 35 + bg_padding),
                     (40, 40, 40), -1)
        cv2.rectangle(annotated, 
                     (text_x - bg_padding, 15 - bg_padding),
                     (text_x + text_size[0] + bg_padding, 35 + bg_padding),
                     (0, 200, 200), 1)
        
        # Draw the plugins text
        cv2.putText(annotated, plugins_text, (text_x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        # No plugins enabled
        no_plugins_text = "No Plugins Enabled"
        text_size = cv2.getTextSize(no_plugins_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (TARGET_WIDTH - text_size[0]) // 2
        
        # Draw background
        bg_padding = 10
        cv2.rectangle(annotated, 
                     (text_x - bg_padding, 15 - bg_padding),
                     (text_x + text_size[0] + bg_padding, 35 + bg_padding),
                     (40, 40, 40), -1)
        
        # Draw the text
        cv2.putText(annotated, no_plugins_text, (text_x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # ----------------------------------------------------
    # RIGHT SIDE: DETECTION STATUS
    # ----------------------------------------------------
    
    # Person detection
    person_x = TARGET_WIDTH - 200
    person_status = "Person: Detected" if kpts is not None else "Person: None"
    person_color = (0, 255, 0) if kpts is not None else (100, 100, 100)
    cv2.putText(annotated, person_status, (person_x, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 1)
    
    # Hand detection
    hand_status = "Hands: Detected" if (hand_res and hand_res.multi_hand_landmarks) else "Hands: None"
    hand_color = (0, 255, 0) if (hand_res and hand_res.multi_hand_landmarks) else (100, 100, 100)
    cv2.putText(annotated, hand_status, (person_x, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
    
    # ====================================================
    # ACTIVE DETECTIONS DISPLAY (Below status bar)
    # ====================================================
    
    # Start Y position for detections
    detection_y = 80
    
    if plugin_msgs:
        # Detection header
        cv2.putText(annotated, "Active Detections:", (10, detection_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        detection_y += 30
        
        # Display each detection message with colored background
        for msg, clr in zip(plugin_msgs, plugin_colors):
            # Draw message with background
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Background rectangle
            cv2.rectangle(annotated, 
                         (5, detection_y - 25),
                         (15 + text_size[0], detection_y + 5),
                         (20, 20, 20), -1)
            cv2.rectangle(annotated,
                         (5, detection_y - 25),
                         (15 + text_size[0], detection_y + 5),
                         clr, 1)
            
            # Message text
            cv2.putText(annotated, msg, (10, detection_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr, 2)
            detection_y += 40
    else:
        # No active detections
        cv2.putText(annotated, "No active detections", (10, detection_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # ====================================================
    # BOTTOM INFO BAR
    # ====================================================
    
    # Draw bottom bar
    cv2.rectangle(annotated, (0, TARGET_HEIGHT - 30), 
                 (TARGET_WIDTH, TARGET_HEIGHT), (30, 30, 30), -1)
    cv2.rectangle(annotated, (0, TARGET_HEIGHT - 30), 
                 (TARGET_WIDTH, TARGET_HEIGHT), (60, 60, 60), 1)
    
    # Frame counter
    cv2.putText(annotated, f"Frame: {frame_count}", 
               (10, TARGET_HEIGHT - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Help text
    help_text = "Press 'Q' to quit | 'S' to screenshot | 'D' for debug"
    help_width = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    cv2.putText(annotated, help_text, 
               (TARGET_WIDTH - help_width - 10, TARGET_HEIGHT - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # ====================================================
    # SHOW WINDOW
    # ====================================================
    cv2.imshow("Aman's Posture Detection System", annotated)
    
    # ====================================================
    # KEYBOARD CONTROLS
    # ====================================================
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n[INFO] Quitting...")
        break
    elif key == ord('s'):
        screenshot_name = f"screenshot_{frame_count}.png"
        cv2.imwrite(screenshot_name, annotated)
        print(f"[INFO] Screenshot saved: {screenshot_name}")
    elif key == ord('d'):
        print(f"\n[DEBUG INFO]")
        print(f"  Frame: {frame_count}")
        print(f"  Device: {DEVICE}")
        print(f"  Plugins enabled: {loaded_plugin_names}")
        print(f"  Active messages: {plugin_msgs}")
        print(f"  Keypoints: {'Present' if kpts is not None else 'None'}")
        print(f"  Hands: {'Detected' if (hand_res and hand_res.multi_hand_landmarks) else 'None'}")

# ============================================================
# CLEANUP
# ============================================================

print("\n" + "="*50)
print("[INFO] Shutting down...")
print("="*50)

cap.release()
hands.close()
cv2.destroyAllWindows()

print(f"\n[INFO] Total frames processed: {frame_count}")

print("[INFO] System shutdown complete. Goodbye!\n")
