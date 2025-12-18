import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import importlib
import torch
import threading
import queue
import time
import base64
import io
from flask import Flask, render_template_string, Response, jsonify
from flask_cors import CORS

# ============================================================
# FLASK APP INITIALIZATION
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIG — ENABLE PLUGINS
# ============================================================

ENABLED_PLUGINS = ["finger_count","raise"]
MODEL_PATH = r"D:\child detections\Child-posture\Project\posture_framework\models\yolov8s-pose.pt"
CAMERA_ID = 0
TARGET_WIDTH = 1024  # Increased for better visibility
TARGET_HEIGHT = 768  # Increased for better visibility

# Global variables for sharing data between threads
current_frame = None
current_messages = []
current_status = {}
frame_queue = queue.Queue(maxsize=2)
is_running = False
detection_thread = None

# ============================================================
# HTML TEMPLATE WITH EMBEDDED CSS AND JAVASCRIPT
# ============================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aman's Posture Detection System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 100%);
            color: #ffffff;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 100%;
            padding: 20px;
        }
        
        /* HEADER STYLES */
        .header {
            text-align: center;
            padding: 30px 20px;
            background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            border: 1px solid #2d46b9;
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #2d46b9, #25d1da);
        }
        
        .main-title {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #2d46b9, #25d1da);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 10px rgba(45, 70, 185, 0.3);
            letter-spacing: 1px;
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: #8a8dff;
            margin-bottom: 20px;
            font-weight: 300;
        }
        
        .status-badges {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        
        .badge {
            padding: 10px 25px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .badge.gpu {
            background: linear-gradient(45deg, #00b09b, #96c93d);
        }
        
        .badge.camera {
            background: linear-gradient(45deg, #3494e6, #ec6ead);
        }
        
        .badge.plugins {
            background: linear-gradient(45deg, #8a2be2, #4a00e0);
        }
        
        /* MAIN CONTENT LAYOUT */
        .content {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        @media (max-width: 1200px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
        
        /* VIDEO FEED SECTION */
        .video-section {
            background: rgba(26, 26, 46, 0.9);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            border: 1px solid #2d46b9;
        }
        
        .section-title {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #8a8dff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .section-title i {
            font-size: 1.8rem;
        }
        
        .video-container {
            position: relative;
            width: 100%;
            height: 600px;
            border-radius: 15px;
            overflow: hidden;
            background: #0a0a0a;
            border: 2px solid #2d46b9;
        }
        
        #video-feed {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        
        .video-overlay {
            position: absolute;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #2d46b9;
        }
        
        .video-stats {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: #cccccc;
        }
        
        /* SIDEBAR SECTION */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 30px;
        }
        
        .info-card {
            background: rgba(26, 26, 46, 0.9);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            border: 1px solid #2d46b9;
        }
        
        .detections-list {
            list-style: none;
            margin-top: 15px;
        }
        
        .detection-item {
            padding: 12px 15px;
            margin-bottom: 10px;
            background: rgba(45, 70, 185, 0.1);
            border-radius: 10px;
            border-left: 4px solid;
            animation: fadeIn 0.5s ease;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .detection-icon {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        
        .plugin-list {
            list-style: none;
            margin-top: 15px;
        }
        
        .plugin-item {
            padding: 10px 15px;
            margin-bottom: 8px;
            background: rgba(138, 141, 255, 0.1);
            border-radius: 8px;
            border-left: 3px solid #8a8dff;
        }
        
        /* CONTROLS SECTION */
        .controls-section {
            background: rgba(26, 26, 46, 0.9);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            border: 1px solid #2d46b9;
        }
        
        .controls-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .control-btn {
            padding: 15px;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .control-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        
        .control-btn.primary {
            background: linear-gradient(45deg, #2d46b9, #25d1da);
            color: white;
        }
        
        .control-btn.secondary {
            background: rgba(45, 70, 185, 0.2);
            color: #8a8dff;
            border: 1px solid #2d46b9;
        }
        
        .control-btn.danger {
            background: linear-gradient(45deg, #ff416c, #ff4b2b);
            color: white;
        }
        
        /* FOOTER */
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 30px;
            color: #666;
            font-size: 0.9rem;
            border-top: 1px solid rgba(45, 70, 185, 0.3);
        }
        
        /* LOADING OVERLAY */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(10, 10, 20, 0.95);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            transition: opacity 0.3s ease;
        }
        
        .loader {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(45, 70, 185, 0.3);
            border-top: 4px solid #2d46b9;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* RESPONSIVE DESIGN */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2.5rem;
            }
            
            .content {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .video-container {
                height: 400px;
            }
            
            .controls-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loader"></div>
        <h2>Initializing Detection System...</h2>
        <p id="loadingStatus">Loading YOLO Model...</p>
    </div>

    <div class="container" id="mainContainer" style="display: none;">
        <!-- HEADER -->
        <div class="header">
            <h1 class="main-title">🦾 Aman's Posture Detection System</h1>
            <p class="subtitle">Advanced AI-powered posture and gesture recognition in real-time</p>
            
            <div class="status-badges">
                <div class="badge gpu" id="gpuStatus">
                    <span>🔌</span>
                    <span>GPU: Loading...</span>
                </div>
                <div class="badge camera" id="cameraStatus">
                    <span>📷</span>
                    <span>Camera: Loading...</span>
                </div>
                <div class="badge plugins" id="pluginsStatus">
                    <span>🔌</span>
                    <span>Plugins: Loading...</span>
                </div>
            </div>
        </div>

        <!-- MAIN CONTENT -->
        <div class="content">
            <!-- VIDEO FEED SECTION -->
            <div class="video-section">
                <h2 class="section-title">
                    <span>📹</span> Live Detection Feed
                </h2>
                <div class="video-container">
                    <img id="video-feed" src="" alt="Live Feed">
                    <div class="video-overlay">
                        <div class="video-stats">
                            <div>FPS: <span id="fpsCounter">0</span></div>
                            <div>Frame: <span id="frameCounter">0</span></div>
                            <div>Resolution: <span id="resolution">1024x768</span></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- SIDEBAR -->
            <div class="sidebar">
                <!-- ACTIVE DETECTIONS -->
                <div class="info-card">
                    <h2 class="section-title">
                        <span>🚨</span> Active Detections
                    </h2>
                    <ul class="detections-list" id="detectionsList">
                        <li class="detection-item" style="border-left-color: #ff4b2b;">
                            <div class="detection-icon" style="background: #ff4b2b;">!</div>
                            <span>Waiting for detection...</span>
                        </li>
                    </ul>
                </div>

                <!-- LOADED PLUGINS -->
                <div class="info-card">
                    <h2 class="section-title">
                        <span>🔧</span> Loaded Plugins
                    </h2>
                    <ul class="plugin-list" id="pluginsList">
                        <li class="plugin-item">Loading plugins...</li>
                    </ul>
                </div>

                <!-- SYSTEM STATUS -->
                <div class="info-card">
                    <h2 class="section-title">
                        <span>📊</span> System Status
                    </h2>
                    <div class="status-info">
                        <p><strong>YOLO Model:</strong> <span id="modelStatus">Loading...</span></p>
                        <p><strong>Hand Detection:</strong> <span id="handStatus">Loading...</span></p>
                        <p><strong>Processing Device:</strong> <span id="deviceStatus">Loading...</span></p>
                        <p><strong>Frame Rate:</strong> <span id="frameRate">0 FPS</span></p>
                    </div>
                </div>
            </div>
        </div>

        <!-- CONTROLS SECTION -->
        <div class="controls-section">
            <h2 class="section-title">
                <span>🎮</span> System Controls
            </h2>
            <div class="controls-grid">
                <button class="control-btn primary" onclick="captureScreenshot()">
                    <span>📸</span> Capture Screenshot
                </button>
                <button class="control-btn secondary" onclick="toggleDebug()">
                    <span>🐛</span> Toggle Debug Info
                </button>
                <button class="control-btn secondary" onclick="toggleFullscreen()">
                    <span>🔍</span> Fullscreen Mode
                </button>
                <button class="control-btn danger" onclick="shutdownSystem()">
                    <span>⏻</span> Shutdown System
                </button>
            </div>
        </div>

        <!-- FOOTER -->
        <div class="footer">
            <p>© 2024 Aman's Posture Detection System | Real-time AI Processing | Built with Flask & OpenCV</p>
            <p style="margin-top: 5px; font-size: 0.8rem;">
                CPU/GPU Usage: <span id="cpuUsage">0%</span> | 
                Memory: <span id="memoryUsage">0 MB</span> |
                Uptime: <span id="uptime">00:00:00</span>
            </p>
        </div>
    </div>

    <script>
        let frameCount = 0;
        let lastTime = Date.now();
        let fps = 0;
        let debugMode = false;
        let startTime = Date.now();
        
        // Update system info
        function updateSystemInfo() {
            const now = Date.now();
            const elapsed = Math.floor((now - startTime) / 1000);
            const hours = Math.floor(elapsed / 3600);
            const minutes = Math.floor((elapsed % 3600) / 60);
            const seconds = elapsed % 60;
            
            document.getElementById('uptime').textContent = 
                `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            
            // Update FPS every second
            const currentTime = Date.now();
            if (currentTime - lastTime >= 1000) {
                fps = frameCount;
                frameCount = 0;
                lastTime = currentTime;
                document.getElementById('fpsCounter').textContent = fps;
            }
            
            // Update frame counter
            document.getElementById('frameCounter').textContent = frameCount;
        }
        
        // Load video feed
        function loadVideoFeed() {
            const videoFeed = document.getElementById('video-feed');
            videoFeed.src = "{{ url_for('video_feed') }}";
            
            videoFeed.onload = function() {
                frameCount++;
                if (debugMode) {
                    console.log(`Frame ${frameCount} loaded`);
                }
            };
            
            videoFeed.onerror = function() {
                console.error('Error loading video feed');
                setTimeout(() => {
                    videoFeed.src = "{{ url_for('video_feed') }}?t=" + new Date().getTime();
                }, 1000);
            };
            
            // Refresh image every 33ms (30fps)
            setInterval(() => {
                videoFeed.src = "{{ url_for('video_feed') }}?t=" + new Date().getTime();
            }, 33);
        }
        
        // Update detection data
        function updateDetectionData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update detections
                    const detectionsList = document.getElementById('detectionsList');
                    if (data.messages && data.messages.length > 0) {
                        detectionsList.innerHTML = '';
                        data.messages.forEach((msg, index) => {
                            const colors = data.colors && data.colors[index] ? data.colors[index] : [255, 75, 43];
                            const color = `rgb(${colors[0]}, ${colors[1]}, ${colors[2]})`;
                            
                            const item = document.createElement('li');
                            item.className = 'detection-item';
                            item.style.borderLeftColor = color;
                            item.innerHTML = `
                                <div class="detection-icon" style="background: ${color}">${index + 1}</div>
                                <span>${msg}</span>
                            `;
                            detectionsList.appendChild(item);
                        });
                    }
                    
                    // Update system status
                    if (data.status) {
                        document.getElementById('gpuStatus').innerHTML = 
                            `<span>🔌</span><span>GPU: ${data.status.device || 'CPU'}</span>`;
                        document.getElementById('cameraStatus').innerHTML = 
                            `<span>📷</span><span>Camera: ${data.status.camera || 'Not connected'}</span>`;
                        document.getElementById('pluginsStatus').innerHTML = 
                            `<span>🔌</span><span>Plugins: ${data.status.plugins_loaded || 0} loaded</span>`;
                        
                        document.getElementById('modelStatus').textContent = data.status.model_loaded || 'Unknown';
                        document.getElementById('handStatus').textContent = data.status.hands_detected ? 'Active' : 'Inactive';
                        document.getElementById('deviceStatus').textContent = data.status.device || 'Unknown';
                        
                        // Update plugins list
                        if (data.status.plugins) {
                            const pluginsList = document.getElementById('pluginsList');
                            pluginsList.innerHTML = '';
                            data.status.plugins.forEach(plugin => {
                                const item = document.createElement('li');
                                item.className = 'plugin-item';
                                item.textContent = plugin;
                                pluginsList.appendChild(item);
                            });
                        }
                    }
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
            
            // Update system info
            updateSystemInfo();
        }
        
        // Control functions
        function captureScreenshot() {
            fetch('/capture')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(`Screenshot saved as: ${data.filename}`);
                    } else {
                        alert('Failed to capture screenshot');
                    }
                });
        }
        
        function toggleDebug() {
            debugMode = !debugMode;
            alert(`Debug mode ${debugMode ? 'enabled' : 'disabled'}`);
        }
        
        function toggleFullscreen() {
            const elem = document.documentElement;
            if (!document.fullscreenElement) {
                if (elem.requestFullscreen) {
                    elem.requestFullscreen();
                } else if (elem.webkitRequestFullscreen) {
                    elem.webkitRequestFullscreen();
                } else if (elem.msRequestFullscreen) {
                    elem.msRequestFullscreen();
                }
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                } else if (document.msExitFullscreen) {
                    document.msExitFullscreen();
                }
            }
        }
        
        function shutdownSystem() {
            if (confirm('Are you sure you want to shutdown the detection system?')) {
                fetch('/shutdown', { method: 'POST' })
                    .then(() => {
                        alert('System shutting down...');
                        setTimeout(() => {
                            window.close();
                        }, 1000);
                    });
            }
        }
        
        // Hide loading overlay when system is ready
        function hideLoadingOverlay() {
            document.getElementById('loadingOverlay').style.opacity = '0';
            setTimeout(() => {
                document.getElementById('loadingOverlay').style.display = 'none';
                document.getElementById('mainContainer').style.display = 'block';
                
                // Initialize video feed
                loadVideoFeed();
                
                // Start updating data
                setInterval(updateDetectionData, 500);
                setInterval(updateSystemInfo, 1000);
                
                // Initial update
                updateDetectionData();
            }, 300);
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            // Check system status and hide loading
            setTimeout(hideLoadingOverlay, 2000);
        });
    </script>
</body>
</html>
'''

# ============================================================
# DETECTION SYSTEM INITIALIZATION
# ============================================================

# Global variables for the detection system
model = None
hands = None
plugins = []
loaded_plugin_names = []
cap = None
is_detection_running = False

def initialize_detection_system():
    """Initialize the detection system components"""
    global model, hands, plugins, loaded_plugin_names, cap
    
    # GPU CHECK
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[INFO] GPU Detected → {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[INFO] Running on CPU")
    
    # LOAD YOLO MODEL
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] YOLO model not found!")
        return False
    
    print("[INFO] Loading YOLO pose model...")
    try:
        model = YOLO(MODEL_PATH)
        model.to(device)
        
        if model.model.yaml.get("kpt_shape", None) is None:
            print("[ERROR] This is NOT a pose model.")
            return False
            
        print("[SUCCESS] YOLO pose model loaded and verified")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        return False
    
    # MEDIAPIPE HANDS INITIALIZATION
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("[INFO] MediaPipe hands initialized")
    
    # LOAD PLUGINS
    PLUGIN_FOLDER = "posture_framework.plugins"
    
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
                print(f"  ✗ {plugin_name}: Module not found")
            except Exception as e:
                print(f"  ✗ {plugin_name}: {e}")
    
    # CAMERA INITIALIZATION
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    
    print(f"[INFO] Camera started at {TARGET_WIDTH}x{TARGET_HEIGHT}")
    return True

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

def detection_loop():
    """Main detection loop running in a separate thread"""
    global current_frame, current_messages, current_status, is_detection_running
    
    frame_count = 0
    
    while is_detection_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Resize frame for processing
        frame_small = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # YOLO POSE DETECTION
        try:
            results = model(frame_small, verbose=False)
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
        
        # HAND DETECTION
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        hand_res = hands.process(rgb)
        
        # Draw hand landmarks
        if hand_res and hand_res.multi_hand_landmarks:
            mp_draw = mp.solutions.drawing_utils
            for hand_landmarks in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    annotated, 
                    hand_landmarks, 
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
                )
        
        # RUN PLUGINS
        plugin_msgs, plugin_colors = run_plugins(frame_small, annotated, kpts, hand_res)
        
        # Add UI overlays
        cv2.putText(annotated, "Live Detection", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated, f"Frame: {frame_count}", (10, TARGET_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Update global variables
        current_frame = annotated
        current_messages = plugin_msgs
        
        # Update status
        current_status = {
            'frame_count': frame_count,
            'person_detected': kpts is not None,
            'hands_detected': hand_res and hand_res.multi_hand_landmarks is not None,
            'plugin_count': len(plugin_msgs),
            'device': 'GPU' if torch.cuda.is_available() else 'CPU',
            'resolution': f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
            'plugins': loaded_plugin_names,
            'plugins_loaded': len(loaded_plugin_names),
            'model_loaded': 'Yes',
            'camera': 'Connected'
        }
        
        # Limit frame rate
        time.sleep(0.03)  # ~30 FPS

def generate_frames():
    """Generate video frames for the web interface"""
    while True:
        if current_frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    """Render the main interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify({
        'messages': current_messages,
        'status': current_status,
        'colors': [[255, 75, 43]] if current_messages else []
    })

@app.route('/capture')
def capture_screenshot():
    """Capture a screenshot"""
    try:
        if current_frame is not None:
            timestamp = int(time.time())
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, current_frame)
            return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
    return jsonify({'success': False})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the detection system"""
    global is_detection_running
    
    is_detection_running = False
    if cap:
        cap.release()
    if hands:
        hands.close()
    
    print("[INFO] System shutdown requested")
    return jsonify({'success': True})

# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    # Print header
    print("\n" + "="*70)
    print("              AMAN'S ADVANCED POSTURE DETECTION SYSTEM")
    print("="*70 + "\n")
    print("🎥 Starting Web Interface...")
    print(f"📡 Server will be available at: http://localhost:5000")
    print("="*70 + "\n")
    
    # Initialize detection system
    if initialize_detection_system():
        # Start detection thread
        is_detection_running = True
        detection_thread = threading.Thread(target=detection_loop, daemon=True)
        detection_thread.start()
        
        # Start Flask server
        print("[INFO] Flask server starting...")
        print("[INFO] Open your browser and navigate to: http://localhost:5000")
        print("\n" + "="*70)
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
        # Cleanup on exit
        is_detection_running = False
        if detection_thread:
            detection_thread.join(timeout=2)
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        
        print("\n[INFO] System shutdown complete. Goodbye!\n")
    else:
        print("[ERROR] Failed to initialize detection system!")
        sys.exit(1)