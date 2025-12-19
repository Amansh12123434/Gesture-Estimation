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
import json
import io
from flask import Flask, render_template_string, Response, jsonify, request
from flask_cors import CORS
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import importlib.util
import glob

# ============================================================
# FLASK APP INITIALIZATION
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# PLUGIN MANAGEMENT SYSTEM
# ============================================================

@dataclass
class PluginInfo:
    """Plugin metadata and state"""
    name: str
    module_name: str
    enabled: bool
    instance: Optional[Any] = None
    display_name: str = ""
    description: str = ""
    version: str = "1.0"
    category: str = "general"
    requires_keypoints: bool = False
    requires_hands: bool = False

class PluginManager:
    """Dynamic plugin manager"""
    
    def __init__(self, plugin_folder="posture_framework.plugins"):
        self.plugin_folder = plugin_folder
        self.available_plugins: Dict[str, PluginInfo] = {}
        self.loaded_plugins: Dict[str, Any] = {}
        self.discover_plugins()
    
    def discover_plugins(self):
        """Discover all available plugins in the plugin folder"""
        try:
            # Try to import the plugin package
            plugin_package = importlib.import_module(self.plugin_folder)
            plugin_path = os.path.dirname(plugin_package.__file__)
            
            # Find all Python files in the plugin directory
            plugin_files = glob.glob(os.path.join(plugin_path, "*.py"))
            
            for plugin_file in plugin_files:
                plugin_name = os.path.splitext(os.path.basename(plugin_file))[0]
                if plugin_name.startswith("__"):
                    continue
                
                try:
                    module = importlib.import_module(f"{self.plugin_folder}.{plugin_name}")
                    if hasattr(module, "Plugin"):
                        plugin_instance = module.Plugin()
                        
                        # Create plugin info
                        plugin_info = PluginInfo(
                            name=plugin_name,
                            module_name=plugin_name,
                            enabled=False,
                            instance=plugin_instance,
                            display_name=getattr(plugin_instance, 'name', plugin_name.replace('_', ' ').title()),
                            description=getattr(plugin_instance, 'description', 'No description available'),
                            version=getattr(plugin_instance, 'version', '1.0'),
                            category=getattr(plugin_instance, 'category', 'general'),
                            requires_keypoints=getattr(plugin_instance, 'requires_keypoints', False),
                            requires_hands=getattr(plugin_instance, 'requires_hands', False)
                        )
                        
                        self.available_plugins[plugin_name] = plugin_info
                        print(f"[PLUGIN] Discovered: {plugin_info.display_name}")
                        
                except Exception as e:
                    print(f"[PLUGIN ERROR] Failed to load {plugin_name}: {e}")
        
        except ImportError:
            print(f"[WARNING] Plugin folder '{self.plugin_folder}' not found")
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin"""
        if plugin_name not in self.available_plugins:
            return False
        
        if not self.available_plugins[plugin_name].enabled:
            self.available_plugins[plugin_name].enabled = True
            self.loaded_plugins[plugin_name] = self.available_plugins[plugin_name].instance
            print(f"[PLUGIN] Enabled: {self.available_plugins[plugin_name].display_name}")
        
        return True
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin"""
        if plugin_name not in self.available_plugins:
            return False
        
        if self.available_plugins[plugin_name].enabled:
            self.available_plugins[plugin_name].enabled = False
            if plugin_name in self.loaded_plugins:
                del self.loaded_plugins[plugin_name]
            print(f"[PLUGIN] Disabled: {self.available_plugins[plugin_name].display_name}")
        
        return True
    
    def toggle_plugin(self, plugin_name: str) -> Tuple[bool, bool]:
        """Toggle plugin state"""
        if plugin_name not in self.available_plugins:
            return False, False
        
        if self.available_plugins[plugin_name].enabled:
            success = self.disable_plugin(plugin_name)
            return success, False  # False = now disabled
        else:
            success = self.enable_plugin(plugin_name)
            return success, True  # True = now enabled
    
    def get_enabled_plugins(self) -> List[str]:
        """Get list of enabled plugin names"""
        return [name for name, plugin in self.available_plugins.items() if plugin.enabled]
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict]:
        """Get plugin information"""
        if plugin_name not in self.available_plugins:
            return None
        
        plugin = self.available_plugins[plugin_name]
        return {
            "name": plugin.name,
            "display_name": plugin.display_name,
            "description": plugin.description,
            "version": plugin.version,
            "category": plugin.category,
            "enabled": plugin.enabled,
            "requires_keypoints": plugin.requires_keypoints,
            "requires_hands": plugin.requires_hands
        }
    
    def get_all_plugins_info(self) -> List[Dict]:
        """Get information for all plugins"""
        return [self.get_plugin_info(name) for name in self.available_plugins]
    
    def run_plugins(self, frame, annotated, kpts, hand_res) -> Tuple[List[str], List[Tuple[int, int, int]]]:
        """Run all enabled plugins"""
        messages, colors = [], []
        
        for plugin_name, plugin_instance in self.loaded_plugins.items():
            try:
                msg, color = plugin_instance.run(frame, annotated, kpts, hand_res)
                
                if msg and color and isinstance(color, (tuple, list)) and len(color) == 3:
                    messages.append(str(msg))
                    colors.append(tuple(int(c) for c in color))
            except Exception as e:
                print(f"[PLUGIN ERROR] {plugin_name}: {e}")
        
        return messages, colors

# ============================================================
# GLOBAL VARIABLES
# ============================================================

MODEL_PATH = r"D:\child detections\Child-posture\Project\posture_framework\models\yolov8s-pose.pt"
CAMERA_ID = 0
TARGET_WIDTH = 1024
TARGET_HEIGHT = 768

# Global variables for sharing data between threads
current_frame = None
current_messages = []
current_status = {}
frame_queue = queue.Queue(maxsize=2)
is_running = False
detection_thread = None

# Initialize plugin manager
plugin_manager = PluginManager()

# Detection system components
model = None
hands = None
cap = None
is_detection_running = False

# ============================================================
# HTML TEMPLATE WITH COMPACT PLUGIN DRAWER
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
            padding: 25px 20px;
            background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
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
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 8px;
            background: linear-gradient(45deg, #2d46b9, #25d1da);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 10px rgba(45, 70, 185, 0.3);
            letter-spacing: 1px;
        }
        
        .subtitle {
            font-size: 1.1rem;
            color: #8a8dff;
            margin-bottom: 15px;
            font-weight: 300;
        }
        
        .status-badges {
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        
        .badge {
            padding: 8px 20px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 6px;
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
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .badge.plugins:hover {
            transform: translateY(-2px);
        }
        
        /* MAIN CONTENT LAYOUT - 2 COLUMNS */
        .content {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 25px;
            margin-bottom: 25px;
        }
        
        @media (max-width: 1200px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
        
        /* VIDEO FEED SECTION */
        .video-section {
            background: rgba(26, 26, 46, 0.9);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
            border: 1px solid #2d46b9;
        }
        
        .section-title {
            font-size: 1.3rem;
            margin-bottom: 15px;
            color: #8a8dff;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .video-container {
            position: relative;
            width: 100%;
            height: 500px;
            border-radius: 12px;
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
            bottom: 15px;
            left: 15px;
            right: 15px;
            background: rgba(0, 0, 0, 0.85);
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #2d46b9;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .video-stats {
            display: flex;
            gap: 20px;
            font-size: 0.85rem;
            color: #cccccc;
        }
        
        .video-controls {
            display: flex;
            gap: 10px;
        }
        
        .video-btn {
            background: rgba(45, 70, 185, 0.3);
            border: 1px solid #2d46b9;
            color: #8a8dff;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.2s;
        }
        
        .video-btn:hover {
            background: rgba(45, 70, 185, 0.5);
            transform: translateY(-1px);
        }
        
        /* RIGHT SIDEBAR */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 25px;
        }
        
        /* COMPACT PLUGINS CARD */
        .plugins-card {
            background: rgba(26, 26, 46, 0.9);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
            border: 1px solid #2d46b9;
        }
        
        .plugins-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .plugins-toggle {
            background: linear-gradient(45deg, #2d46b9, #25d1da);
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 5px;
            transition: all 0.2s;
        }
        
        .plugins-toggle:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(45, 70, 185, 0.4);
        }
        
        .compact-plugins-list {
            list-style: none;
            max-height: 150px;
            overflow-y: auto;
            padding-right: 5px;
        }
        
        .compact-plugins-list::-webkit-scrollbar {
            width: 6px;
        }
        
        .compact-plugins-list::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        
        .compact-plugins-list::-webkit-scrollbar-thumb {
            background: #2d46b9;
            border-radius: 3px;
        }
        
        .compact-plugin-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: rgba(138, 141, 255, 0.1);
            border-radius: 8px;
            border-left: 3px solid #8a8dff;
            transition: all 0.2s;
        }
        
        .compact-plugin-item:hover {
            background: rgba(138, 141, 255, 0.2);
        }
        
        .compact-plugin-name {
            font-weight: 600;
            font-size: 0.95rem;
            color: #8a8dff;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 150px;
        }
        
        .compact-plugin-toggle {
            padding: 4px 12px;
            border: none;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            min-width: 70px;
            text-align: center;
        }
        
        .plugin-enabled {
            background: linear-gradient(45deg, #00b09b, #96c93d);
            color: white;
        }
        
        .plugin-disabled {
            background: rgba(255, 255, 255, 0.1);
            color: #ccc;
            border: 1px solid #666;
        }
        
        /* DETECTIONS CARD */
        .detections-card {
            background: rgba(26, 26, 46, 0.9);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
            border: 1px solid #2d46b9;
            flex-grow: 1;
        }
        
        .detections-list {
            list-style: none;
            margin-top: 10px;
            max-height: 180px;
            overflow-y: auto;
            padding-right: 5px;
        }
        
        .detection-item {
            padding: 10px 12px;
            margin-bottom: 8px;
            background: rgba(45, 70, 185, 0.1);
            border-radius: 8px;
            border-left: 4px solid;
            animation: fadeIn 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .detection-icon {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.8rem;
            flex-shrink: 0;
        }
        
        /* PLUGIN DRAWER (MODAL) */
        .plugin-drawer {
            position: fixed;
            top: 0;
            right: -450px;
            width: 400px;
            height: 100vh;
            background: rgba(26, 26, 46, 0.95);
            backdrop-filter: blur(10px);
            z-index: 1000;
            transition: right 0.3s ease;
            box-shadow: -5px 0 30px rgba(0, 0, 0, 0.5);
            border-left: 1px solid #2d46b9;
            display: flex;
            flex-direction: column;
        }
        
        .plugin-drawer.active {
            right: 0;
        }
        
        .drawer-header {
            padding: 25px;
            background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
            border-bottom: 1px solid #2d46b9;
        }
        
        .drawer-title {
            font-size: 1.5rem;
            color: #8a8dff;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .plugin-search {
            width: 100%;
            padding: 10px 15px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid #2d46b9;
            border-radius: 8px;
            color: white;
            font-size: 0.9rem;
        }
        
        .plugin-search:focus {
            outline: none;
            border-color: #25d1da;
        }
        
        .drawer-content {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .drawer-close {
            position: absolute;
            top: 15px;
            right: 15px;
            background: none;
            border: none;
            color: #8a8dff;
            font-size: 1.5rem;
            cursor: pointer;
            padding: 5px;
            border-radius: 5px;
        }
        
        .drawer-close:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .plugin-detail-item {
            background: rgba(138, 141, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 3px solid #8a8dff;
        }
        
        .plugin-detail-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 10px;
        }
        
        .plugin-detail-name {
            font-size: 1.1rem;
            font-weight: 600;
            color: #8a8dff;
        }
        
        .plugin-detail-description {
            font-size: 0.9rem;
            color: #ccc;
            margin-bottom: 10px;
            line-height: 1.4;
        }
        
        .plugin-detail-meta {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        
        .plugin-meta-tag {
            background: rgba(255, 255, 255, 0.1);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            color: #999;
        }
        
        /* CONTROLS SECTION */
        .controls-section {
            background: rgba(26, 26, 46, 0.9);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
            border: 1px solid #2d46b9;
        }
        
        .controls-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-top: 15px;
        }
        
        @media (max-width: 768px) {
            .controls-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        .control-btn {
            padding: 12px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        
        .control-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
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
            padding: 15px;
            margin-top: 25px;
            color: #666;
            font-size: 0.85rem;
            border-top: 1px solid rgba(45, 70, 185, 0.3);
        }
        
        .system-stats {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 8px;
            font-size: 0.8rem;
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
            z-index: 2000;
            transition: opacity 0.3s ease;
        }
        
        .loader {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(45, 70, 185, 0.3);
            border-top: 3px solid #2d46b9;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* OVERLAY FOR DRAWER */
        .overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            z-index: 999;
            display: none;
            backdrop-filter: blur(3px);
        }
        
        .overlay.active {
            display: block;
        }
        
        /* RESPONSIVE DESIGN */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2rem;
            }
            
            .content {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .video-container {
                height: 350px;
            }
            
            .plugin-drawer {
                width: 100%;
                right: -100%;
            }
            
            .controls-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <!-- LOADING OVERLAY -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loader"></div>
        <h2>Initializing Detection System...</h2>
        <p id="loadingStatus">Loading Plugins...</p>
    </div>
    
    <!-- PLUGIN DRAWER -->
    <div class="overlay" id="drawerOverlay" onclick="closePluginDrawer()"></div>
    <div class="plugin-drawer" id="pluginDrawer">
        <button class="drawer-close" onclick="closePluginDrawer()">×</button>
        <div class="drawer-header">
            <h2 class="drawer-title">
                <span>🔌</span> Plugin Manager
            </h2>
            <input type="text" 
                   class="plugin-search" 
                   id="pluginSearch" 
                   placeholder="Search plugins by name or description..."
                   onkeyup="searchPlugins()">
        </div>
        <div class="drawer-content" id="drawerContent">
            <div id="pluginDetailsList">
                <!-- Plugin details will be loaded here -->
            </div>
        </div>
    </div>
    
    <!-- MAIN CONTAINER -->
    <div class="container" id="mainContainer" style="display: none;">
        <!-- HEADER -->
        <div class="header">
            <h1 class="main-title">🦾 Aman's Posture Detection</h1>
            <p class="subtitle">Real-time AI with dynamic plugin management</p>
            
            <div class="status-badges">
                <div class="badge gpu" id="gpuStatus">
                    <span>🔌</span>
                    <span>GPU: Loading...</span>
                </div>
                <div class="badge camera" id="cameraStatus">
                    <span>📷</span>
                    <span>Camera: Loading...</span>
                </div>
                <div class="badge plugins" id="pluginsStatus" onclick="openPluginDrawer()">
                    <span>🔌</span>
                    <span>Plugins: 0 active</span>
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
                            <div>Detections: <span id="activeDetections">0</span></div>
                        </div>
                        <div class="video-controls">
                            <button class="video-btn" onclick="toggleFullscreen()">
                                <span>⛶</span> Fullscreen
                            </button>
                            <button class="video-btn" onclick="captureScreenshot()">
                                <span>📸</span> Screenshot
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- SIDEBAR -->
            <div class="sidebar">
                <!-- PLUGINS CARD -->
                <div class="plugins-card">
                    <div class="plugins-header">
                        <h2 class="section-title">
                            <span>🔌</span> Active Plugins
                        </h2>
                        <button class="plugins-toggle" onclick="openPluginDrawer()">
                            <span>⚙️</span> Manage
                        </button>
                    </div>
                    <ul class="compact-plugins-list" id="compactPluginsList">
                        <li class="compact-plugin-item">
                            <div class="compact-plugin-name">Loading plugins...</div>
                            <button class="compact-plugin-toggle plugin-disabled" disabled>...</button>
                        </li>
                    </ul>
                </div>

                <!-- DETECTIONS CARD -->
                <div class="detections-card">
                    <h2 class="section-title">
                        <span>🚨</span> Active Detections
                    </h2>
                    <ul class="detections-list" id="detectionsList">
                        <li class="detection-item" style="border-left-color: #666;">
                            <div class="detection-icon" style="background: #666;">!</div>
                            <span>Waiting for detection...</span>
                        </li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- CONTROLS SECTION -->
        <div class="controls-section">
            <h2 class="section-title">
                <span>🎮</span> System Controls
            </h2>
            <div class="controls-grid">
                <button class="control-btn primary" onclick="refreshSystem()">
                    <span>🔄</span> Refresh
                </button>
                <button class="control-btn secondary" onclick="toggleAllPlugins()">
                    <span>⚡</span> Toggle All
                </button>
                <button class="control-btn secondary" onclick="toggleDebug()">
                    <span>🐛</span> Debug
                </button>
                <button class="control-btn danger" onclick="shutdownSystem()">
                    <span>⏻</span> Shutdown
                </button>
            </div>
        </div>

        <!-- FOOTER -->
        <div class="footer">
            <p>© 2024 Aman's Posture Detection System | Real-time AI Processing | Dynamic Plugin Management</p>
            <div class="system-stats">
                <span>CPU/GPU: <span id="cpuUsage">0%</span></span>
                <span>Memory: <span id="memoryUsage">0 MB</span></span>
                <span>Uptime: <span id="uptime">00:00:00</span></span>
                <span>Plugins: <span id="totalPlugins">0</span>/<span id="enabledPlugins">0</span></span>
            </div>
        </div>
    </div>

    <script>
        let frameCount = 0;
        let lastTime = Date.now();
        let fps = 0;
        let debugMode = false;
        let startTime = Date.now();
        let allPlugins = [];
        let drawerOpen = false;
        
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
        
        // Load and display plugins
        function loadPlugins() {
            fetch('/api/plugins')
                .then(response => response.json())
                .then(data => {
                    allPlugins = data.plugins || [];
                    updateCompactPluginsList(allPlugins);
                    updatePluginStats(allPlugins);
                })
                .catch(error => {
                    console.error('Error loading plugins:', error);
                });
        }
        
        // Update compact plugins list
        function updateCompactPluginsList(plugins) {
            const compactList = document.getElementById('compactPluginsList');
            const enabledPlugins = plugins.filter(p => p.enabled);
            
            if (enabledPlugins.length === 0) {
                compactList.innerHTML = `
                    <li class="compact-plugin-item">
                        <div class="compact-plugin-name">No active plugins</div>
                        <button class="compact-plugin-toggle plugin-disabled" 
                                onclick="openPluginDrawer()" 
                                style="cursor: pointer;">
                            Add
                        </button>
                    </li>
                `;
                return;
            }
            
            compactList.innerHTML = '';
            
            // Show only first 5 enabled plugins
            enabledPlugins.slice(0, 5).forEach(plugin => {
                const item = document.createElement('li');
                item.className = 'compact-plugin-item';
                item.innerHTML = `
                    <div class="compact-plugin-name" title="${plugin.display_name}">
                        ${plugin.display_name}
                    </div>
                    <button class="compact-plugin-toggle plugin-enabled" 
                            onclick="togglePlugin('${plugin.name}')"
                            title="Click to disable">
                        ON
                    </button>
                `;
                compactList.appendChild(item);
            });
            
            // Show "more" indicator if there are more plugins
            if (enabledPlugins.length > 5) {
                const moreItem = document.createElement('li');
                moreItem.className = 'compact-plugin-item';
                moreItem.innerHTML = `
                    <div class="compact-plugin-name">
                        +${enabledPlugins.length - 5} more...
                    </div>
                    <button class="compact-plugin-toggle plugin-disabled" 
                            onclick="openPluginDrawer()"
                            style="cursor: pointer;">
                        View
                    </button>
                `;
                compactList.appendChild(moreItem);
            }
        }
        
        // Open plugin drawer
        function openPluginDrawer() {
            drawerOpen = true;
            document.getElementById('pluginDrawer').classList.add('active');
            document.getElementById('drawerOverlay').classList.add('active');
            loadPluginDrawerDetails();
        }
        
        // Close plugin drawer
        function closePluginDrawer() {
            drawerOpen = false;
            document.getElementById('pluginDrawer').classList.remove('active');
            document.getElementById('drawerOverlay').classList.remove('active');
        }
        
        // Load plugin details for drawer
        function loadPluginDrawerDetails() {
            const detailsList = document.getElementById('pluginDetailsList');
            
            if (allPlugins.length === 0) {
                detailsList.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">No plugins found</p>';
                return;
            }
            
            // Sort plugins: enabled first, then by name
            const sortedPlugins = [...allPlugins].sort((a, b) => {
                if (a.enabled !== b.enabled) return b.enabled - a.enabled;
                return a.display_name.localeCompare(b.display_name);
            });
            
            detailsList.innerHTML = '';
            
            sortedPlugins.forEach(plugin => {
                const detailItem = document.createElement('div');
                detailItem.className = 'plugin-detail-item';
                
                const toggleText = plugin.enabled ? 'Disable' : 'Enable';
                const toggleClass = plugin.enabled ? 'plugin-enabled' : 'plugin-disabled';
                
                detailItem.innerHTML = `
                    <div class="plugin-detail-header">
                        <div class="plugin-detail-name">${plugin.display_name}</div>
                        <button class="compact-plugin-toggle ${toggleClass}" 
                                onclick="togglePlugin('${plugin.name}')"
                                style="min-width: 80px;">
                            ${toggleText}
                        </button>
                    </div>
                    <div class="plugin-detail-description">${plugin.description}</div>
                    <div class="plugin-detail-meta">
                        <span class="plugin-meta-tag">v${plugin.version}</span>
                        <span class="plugin-meta-tag">${plugin.category}</span>
                        ${plugin.requires_keypoints ? '<span class="plugin-meta-tag">📐 Keypoints</span>' : ''}
                        ${plugin.requires_hands ? '<span class="plugin-meta-tag">✋ Hands</span>' : ''}
                        <span class="plugin-meta-tag">${plugin.enabled ? '🟢 Active' : '⚫ Inactive'}</span>
                    </div>
                `;
                
                detailsList.appendChild(detailItem);
            });
        }
        
        // Search plugins in drawer
        function searchPlugins() {
            const searchTerm = document.getElementById('pluginSearch').value.toLowerCase();
            const detailsList = document.getElementById('pluginDetailsList');
            
            if (!searchTerm) {
                loadPluginDrawerDetails();
                return;
            }
            
            const filteredPlugins = allPlugins.filter(plugin => 
                plugin.display_name.toLowerCase().includes(searchTerm) ||
                plugin.description.toLowerCase().includes(searchTerm) ||
                plugin.category.toLowerCase().includes(searchTerm)
            );
            
            if (filteredPlugins.length === 0) {
                detailsList.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">No plugins found matching "' + searchTerm + '"</p>';
                return;
            }
            
            detailsList.innerHTML = '';
            
            filteredPlugins.forEach(plugin => {
                const detailItem = document.createElement('div');
                detailItem.className = 'plugin-detail-item';
                
                const toggleText = plugin.enabled ? 'Disable' : 'Enable';
                const toggleClass = plugin.enabled ? 'plugin-enabled' : 'plugin-disabled';
                
                detailItem.innerHTML = `
                    <div class="plugin-detail-header">
                        <div class="plugin-detail-name">${plugin.display_name}</div>
                        <button class="compact-plugin-toggle ${toggleClass}" 
                                onclick="togglePlugin('${plugin.name}')"
                                style="min-width: 80px;">
                            ${toggleText}
                        </button>
                    </div>
                    <div class="plugin-detail-description">${plugin.description}</div>
                    <div class="plugin-detail-meta">
                        <span class="plugin-meta-tag">v${plugin.version}</span>
                        <span class="plugin-meta-tag">${plugin.category}</span>
                        ${plugin.requires_keypoints ? '<span class="plugin-meta-tag">📐 Keypoints</span>' : ''}
                        ${plugin.requires_hands ? '<span class="plugin-meta-tag">✋ Hands</span>' : ''}
                    </div>
                `;
                
                detailsList.appendChild(detailItem);
            });
        }
        
        // Toggle plugin state
        function togglePlugin(pluginName) {
            fetch('/api/plugins/toggle', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ plugin_name: pluginName })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update the specific plugin in allPlugins array
                    const pluginIndex = allPlugins.findIndex(p => p.name === pluginName);
                    if (pluginIndex !== -1) {
                        allPlugins[pluginIndex].enabled = data.enabled;
                    }
                    
                    // Update all displays
                    updateCompactPluginsList(allPlugins);
                    updatePluginStats(allPlugins);
                    
                    // Refresh drawer if open
                    if (drawerOpen) {
                        loadPluginDrawerDetails();
                    }
                    
                    // Refresh detections
                    updateDetectionData();
                } else {
                    alert('Failed to toggle plugin: ' + (data.message || 'Unknown error'));
                }
            })
            .catch(error => {
                console.error('Error toggling plugin:', error);
                alert('Error toggling plugin');
            });
        }
        
        // Toggle all plugins
        function toggleAllPlugins() {
            const action = confirm('Enable all plugins? Cancel to disable all.');
            const enable = action === true;
            
            fetch('/api/plugins/toggle_all', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ enable: enable })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update all plugins in the array
                    allPlugins.forEach(plugin => {
                        plugin.enabled = enable;
                    });
                    
                    // Update all displays
                    updateCompactPluginsList(allPlugins);
                    updatePluginStats(allPlugins);
                    
                    // Refresh drawer if open
                    if (drawerOpen) {
                        loadPluginDrawerDetails();
                    }
                    
                    // Refresh detections
                    updateDetectionData();
                } else {
                    alert('Failed to toggle plugins');
                }
            })
            .catch(error => {
                console.error('Error toggling all plugins:', error);
                alert('Error toggling plugins');
            });
        }
        
        // Update plugin statistics
        function updatePluginStats(plugins) {
            const total = plugins.length;
            const enabled = plugins.filter(p => p.enabled).length;
            
            document.getElementById('totalPlugins').textContent = total;
            document.getElementById('enabledPlugins').textContent = enabled;
            document.getElementById('pluginsStatus').innerHTML = 
                `<span>🔌</span><span>Plugins: ${enabled}/${total} active</span>`;
        }
        
        // Update detection data
        function updateDetectionData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update detections count
                    const detectionCount = data.messages ? data.messages.length : 0;
                    document.getElementById('activeDetections').textContent = detectionCount;
                    
                    // Update detections list
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
                    } else {
                        detectionsList.innerHTML = `
                            <li class="detection-item" style="border-left-color: #666;">
                                <div class="detection-icon" style="background: #666;">!</div>
                                <span>No active detections</span>
                            </li>
                        `;
                    }
                    
                    // Update system status
                    if (data.status) {
                        document.getElementById('gpuStatus').innerHTML = 
                            `<span>🔌</span><span>GPU: ${data.status.device || 'CPU'}</span>`;
                        document.getElementById('cameraStatus').innerHTML = 
                            `<span>📷</span><span>Camera: ${data.status.camera || 'Not connected'}</span>`;
                    }
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
            
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
        
        function refreshSystem() {
            loadPlugins();
            updateDetectionData();
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
                
                // Initialize system
                loadVideoFeed();
                loadPlugins();
                
                // Start updating data
                setInterval(updateDetectionData, 500);
                setInterval(updateSystemInfo, 1000);
                
                // Initial update
                updateDetectionData();
            }, 300);
        }
        
        // Close drawer with ESC key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && drawerOpen) {
                closePluginDrawer();
            }
        });
        
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

def initialize_detection_system():
    """Initialize the detection system components"""
    global model, hands, cap
    
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
    
    # CAMERA INITIALIZATION
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    
    print(f"[INFO] Camera started at {TARGET_WIDTH}x{TARGET_HEIGHT}")
    return True

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
        
        # RUN ENABLED PLUGINS
        plugin_msgs, plugin_colors = plugin_manager.run_plugins(frame_small, annotated, kpts, hand_res)
        
        # Add UI overlays
        cv2.putText(annotated, "Live Detection", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated, f"Frame: {frame_count}", (10, TARGET_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(annotated, f"Plugins: {len(plugin_manager.get_enabled_plugins())} active", 
                    (10, TARGET_HEIGHT - 40),
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
            'plugins': plugin_manager.get_enabled_plugins(),
            'plugins_loaded': len(plugin_manager.get_enabled_plugins()),
            'total_plugins': len(plugin_manager.available_plugins),
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

@app.route('/api/plugins')
def get_plugins():
    """Get all available plugins"""
    plugins_info = plugin_manager.get_all_plugins_info()
    return jsonify({
        'plugins': plugins_info,
        'total': len(plugins_info),
        'enabled': len(plugin_manager.get_enabled_plugins())
    })

@app.route('/api/plugins/toggle', methods=['POST'])
def toggle_plugin():
    """Toggle a plugin on/off"""
    data = request.json
    plugin_name = data.get('plugin_name')
    
    if not plugin_name:
        return jsonify({'success': False, 'message': 'Plugin name required'})
    
    success, enabled = plugin_manager.toggle_plugin(plugin_name)
    
    if success:
        return jsonify({
            'success': True,
            'enabled': enabled,
            'message': f'Plugin {"enabled" if enabled else "disabled"} successfully'
        })
    else:
        return jsonify({'success': False, 'message': 'Plugin not found'})

@app.route('/api/plugins/toggle_all', methods=['POST'])
def toggle_all_plugins():
    """Toggle all plugins on or off"""
    data = request.json
    enable = data.get('enable', False)
    
    # Get all plugin names
    plugin_names = list(plugin_manager.available_plugins.keys())
    
    success_count = 0
    for plugin_name in plugin_names:
        if enable:
            success = plugin_manager.enable_plugin(plugin_name)
        else:
            success = plugin_manager.disable_plugin(plugin_name)
        
        if success:
            success_count += 1
    
    return jsonify({
        'success': success_count > 0,
        'count': success_count,
        'enabled': enable
    })

@app.route('/api/plugins/<plugin_name>')
def get_plugin_info(plugin_name):
    """Get specific plugin information"""
    info = plugin_manager.get_plugin_info(plugin_name)
    
    if info:
        return jsonify({'success': True, 'plugin': info})
    else:
        return jsonify({'success': False, 'message': 'Plugin not found'})

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
    print("         AMAN'S POSTURE DETECTION SYSTEM WITH PLUGIN MANAGER")
    print("="*70 + "\n")
    
    print("[INFO] Discovering plugins...")
    print(f"[INFO] Found {len(plugin_manager.available_plugins)} available plugins")
    
    # Enable some plugins by default for demo
    default_plugins = ["finger_count", "raise"]
    for plugin_name in default_plugins:
        if plugin_name in plugin_manager.available_plugins:
            plugin_manager.enable_plugin(plugin_name)
            print(f"[INFO] Enabled default plugin: {plugin_name}")
    
    print(f"[INFO] {len(plugin_manager.get_enabled_plugins())} plugins enabled")
    
    # Initialize detection system
    if initialize_detection_system():
        # Start detection thread
        is_detection_running = True
        detection_thread = threading.Thread(target=detection_loop, daemon=True)
        detection_thread.start()
        
        # Start Flask server
        print("\n" + "="*70)
        print("[INFO] Flask server starting...")
        print("[INFO] Open your browser and navigate to: http://localhost:5000")
        print("\n[INFO] Interface Features:")
        print("  • Clean, compact design with more video space")
        print("  • Plugin manager slide-out drawer (click Plugin badge)")
        print("  • Quick plugin toggle in compact list")
        print("  • Fullscreen mode button")
        print("  • Real-time plugin management")
        print("="*70 + "\n")
        
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