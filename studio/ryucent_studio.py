import sys
import os
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "raw-images")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "images")
SUPABASE_AUTH_TABLE = os.getenv("SUPABASE_AUTH_TABLE", "terminal_access")
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QListWidget, 
                             QListWidgetItem, QGraphicsView, QGraphicsScene, 
                             QGraphicsPixmapItem, QGraphicsRectItem, QFrame,
                             QSplitter, QStatusBar, QMessageBox,
                             QLineEdit, QDialog)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, Slot
from PySide6.QtGui import QPixmap, QImage, QColor, QPen, QBrush, QIcon, QKeySequence, QShortcut, QPainter
from pathlib import Path
import requests
from io import BytesIO

# Resolve paths relative to script location
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent

# --- CONFIG ---
# Configuration is pulled from .env file

# --- STYLING ---
DARK_THEME = """
QMainWindow { background-color: #121212; }
QWidget { background-color: #121212; color: #E0E0E0; font-family: 'Segoe UI', Arial; }
QFrame#Sidebar { background-color: #1E1E1E; border-right: 1px solid #333; }
QFrame#Gallery { background-color: #1E1E1E; border-left: 1px solid #333; }
QListWidget { background-color: #1E1E1E; border: none; outline: none; }
QListWidget::item { padding: 10px; border-bottom: 1px solid #2A2A2A; }
QListWidget::item:selected { background-color: #FF6B00; color: white; border-radius: 4px; }
QPushButton { 
    background-color: #333; border: none; padding: 8px 15px; 
    border-radius: 4px; font-weight: bold; 
}
QPushButton:hover { background-color: #444; }
QPushButton#SubmitBtn { background-color: #FF6B00; color: white; font-size: 14px; }
QPushButton#SubmitBtn:hover { background-color: #FF8533; }
QPushButton#DeleteBtn { background-color: #D32F2F; color: white; }
QLabel#Header { font-size: 18px; font-weight: bold; color: #FF6B00; margin-bottom: 10px; }
QLabel#Instructions { color: #888; font-size: 12px; line-height: 1.5; }
QLineEdit { 
    background-color: #2A2A2A; border: 1px solid #444; padding: 10px; 
    border-radius: 4px; color: white; font-size: 14px;
}
QLineEdit:focus { border: 1px solid #FF6B00; }
QLabel#LoginTitle { font-size: 24px; font-weight: bold; color: white; margin-bottom: 20px; }
"""

class LoginDialog(QWidget):
    authenticated = Signal(bool)

    def __init__(self, supabase):
        super().__init__()
        self.supabase = supabase
        self.setWindowTitle("Ryucent Login")
        self.setFixedSize(400, 300)
        self.setStyleSheet(DARK_THEME)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(15)

        title = QLabel("RYUCENT STUDIO")
        title.setObjectName("LoginTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Enter Access Key...")
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.returnPressed.connect(self.check_auth)
        layout.addWidget(self.key_input)

        self.login_btn = QPushButton("LOGIN")
        self.login_btn.setObjectName("SubmitBtn")
        self.login_btn.setFixedHeight(45)
        self.login_btn.clicked.connect(self.check_auth)
        layout.addWidget(self.login_btn)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #D32F2F; font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        exit_btn = QPushButton("EXIT")
        exit_btn.setStyleSheet("background: transparent; color: #666;")
        exit_btn.clicked.connect(sys.exit)
        layout.addWidget(exit_btn)

    def check_auth(self):
        access_key = self.key_input.text().strip()
        if not access_key:
            self.status_label.setText("Please enter a key")
            return

        self.status_label.setText("Authenticating...")
        self.status_label.setStyleSheet("color: #FF6B00;")
        QApplication.processEvents()

        try:
            # Check terminal_access table for the key
            response = self.supabase.table(SUPABASE_AUTH_TABLE).select("*").eq("key", access_key).limit(1).execute()
            docs = response.data
            
            if len(docs) > 0 and docs[0].get("active", False):
                self.authenticated.emit(True)
                self.close()
            else:
                self.status_label.setText("Invalid or Expired Key")
                self.status_label.setStyleSheet("color: #D32F2F;")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color: #D32F2F;")

class LabelItem(QGraphicsRectItem):
    def __init__(self, rect, parent=None):
        super().__init__(rect, parent)
        self.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
        self.setBrush(QBrush(QColor(0, 255, 0, 40)))
        self.setFlags(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsRectItem.GraphicsItemFlag.ItemIsFocusable)

class ImageCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        self.pixmap_item = None
        self.current_boxes = []
        self.drawing = False
        self.start_point = None
        self.temp_rect = None
        
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

    def set_image(self, pixmap):
        self.current_boxes = [] # Reset tracking list
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.drawing = True
            self.start_point = self.mapToScene(event.pos())
            self.temp_rect = QGraphicsRectItem()
            self.temp_rect.setPen(QPen(QColor(0, 255, 0), 2))
            self.scene.addItem(self.temp_rect)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing:
            end_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, end_point).normalized()
            self.temp_rect.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            rect = self.temp_rect.rect()
            self.scene.removeItem(self.temp_rect)
            if rect.width() > 5 and rect.height() > 5:
                box = LabelItem(rect)
                self.scene.addItem(box)
                self.current_boxes.append(box)
        else:
            super().mouseReleaseEvent(event)

    def clear_boxes(self):
        for box in self.current_boxes:
            try:
                if box.scene() == self.scene:
                    self.scene.removeItem(box)
            except RuntimeError:
                pass # Already deleted
        self.current_boxes = []

    def get_yolo_labels(self):
        if not self.pixmap_item: return []
        w = self.pixmap_item.pixmap().width()
        h = self.pixmap_item.pixmap().height()
        labels = []
        for box in self.current_boxes:
            rect = box.rect()
            x_center = (rect.x() + rect.width()/2) / w
            y_center = (rect.y() + rect.height()/2) / h
            bw = rect.width() / w
            bh = rect.height() / h
            labels.append({"class": 0, "rel_coords": [x_center, y_center, bw, bh]})
        return labels

class RyucentStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ryucent Labeling Studio")
        self.resize(1400, 900)
        self.setStyleSheet(DARK_THEME)
        
        # Supabase Init
        if not SUPABASE_URL or not SUPABASE_KEY:
            QMessageBox.critical(self, "Error", "Supabase credentials not found in .env!")
            sys.exit(1)
        
        try:
            self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("[INFO] Connected to Supabase.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect to Supabase: {e}")
            sys.exit(1)
        
        self.init_ui()
        self.load_data("pending")

    def init_ui(self):
        # Initialize canvas early to allow signal connections
        self.canvas = ImageCanvas()
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 1. Left Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(sidebar)
        
        logo_label = QLabel("RYUCENT STUDIO")
        logo_label.setObjectName("Header")
        sidebar_layout.addWidget(logo_label)
        
        instr = QLabel("<b>CONTROLS:</b><br>"
                      "• Drag L-Click: Draw Box<br>"
                      "• Mouse Wheel: Zoom<br>"
                      "• Middle Click: Pan<br>"
                      "• [S]: Submit & Next<br>"
                      "• [C]: Clear Canvas<br>"
                      "• [Del]: Delete Image")
        instr.setObjectName("Instructions")
        instr.setWordWrap(True)
        sidebar_layout.addWidget(instr)
        
        sidebar_layout.addStretch()
        
        self.mode_btn = QPushButton("VIEW: PENDING")
        self.mode_btn.clicked.connect(self.toggle_mode)
        sidebar_layout.addWidget(self.mode_btn)
        
        self.clear_btn = QPushButton("Clear Current")
        self.clear_btn.clicked.connect(self.canvas.clear_boxes)
        sidebar_layout.addWidget(self.clear_btn)

        layout.addWidget(sidebar)

        # 2. Main Canvas Area
        center_layout = QVBoxLayout()
        center_layout.addWidget(self.canvas)
        
        bottom_bar = QFrame()
        bottom_layout = QHBoxLayout(bottom_bar)
        
        self.submit_btn = QPushButton("SUBMIT LABELS [S]")
        self.submit_btn.setObjectName("SubmitBtn")
        self.submit_btn.clicked.connect(self.submit_labels)
        bottom_layout.addWidget(self.submit_btn)
        
        self.delete_btn = QPushButton("DELETE IMAGE")
        self.delete_btn.setObjectName("DeleteBtn")
        self.delete_btn.clicked.connect(self.delete_image)
        bottom_layout.addWidget(self.delete_btn)
        
        center_layout.addWidget(bottom_bar)
        layout.addLayout(center_layout, 1)

        # 3. Right Gallery
        gallery_frame = QFrame()
        gallery_frame.setObjectName("Gallery")
        gallery_frame.setFixedWidth(300)
        gallery_layout = QVBoxLayout(gallery_frame)
        
        gallery_layout.addWidget(QLabel("CLOUD GALLERY"))
        self.img_list = QListWidget()
        self.img_list.itemClicked.connect(self.on_item_clicked)
        gallery_layout.addWidget(self.img_list)
        
        layout.addWidget(gallery_frame)

        # Shortcuts
        QShortcut(QKeySequence("S"), self, self.submit_labels)
        QShortcut(QKeySequence("C"), self, self.canvas.clear_boxes)

        self.current_mode = "pending"
        self.current_doc = None

    def toggle_mode(self):
        self.current_mode = "labeled" if self.current_mode == "pending" else "pending"
        self.mode_btn.setText(f"VIEW: {self.current_mode.upper()}")
        self.load_data(self.current_mode)

    def load_data(self, status):
        self.img_list.clear()
        try:
            response = self.supabase.table(SUPABASE_TABLE).select("*").eq("status", status).limit(100).execute()
            docs = response.data
            for data in docs:
                item = QListWidgetItem(data.get("filename"))
                item.setData(Qt.ItemDataRole.UserRole, data.get("filename")) # Using filename as ID if no UUID
                item.setData(Qt.ItemDataRole.UserRole + 1, data.get("storage_path"))
                item.setData(Qt.ItemDataRole.UserRole + 2, data) # Store full record
                self.img_list.addItem(item)
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
        
        if self.img_list.count() > 0:
            self.img_list.setCurrentRow(0)
            self.on_item_clicked(self.img_list.item(0))

    def on_item_clicked(self, item):
        filename = item.data(Qt.ItemDataRole.UserRole)
        storage_path = item.data(Qt.ItemDataRole.UserRole + 1)
        self.current_doc = filename
        
        # Download image from Supabase
        try:
            img_data = self.supabase.storage.from_(SUPABASE_BUCKET).download(storage_path)
            
            pixmap = QPixmap()
            pixmap.loadFromData(img_data)
        except Exception as e:
            print(f"[ERROR] Failed to download image: {e}")
            self.statusBar().showMessage(f"Download Error: {e}", 5000)
            return

        # Clear canvas before loading new image/boxes
        self.canvas.clear_boxes()
        self.canvas.set_image(pixmap)
        
        # If in labeled mode, draw existing boxes
        if self.current_mode == "labeled":
            data = item.data(Qt.ItemDataRole.UserRole + 2)
            for lbl in data.get("labels", []):
                coords = lbl["rel_coords"]
                w, h = pixmap.width(), pixmap.height()
                x_center, y_center, bw, bh = coords
                x1 = (x_center - bw/2) * w
                y1 = (y_center - bh/2) * h
                rect = QRectF(x1, y1, bw*w, bh*h)
                box = LabelItem(rect)
                self.canvas.scene.addItem(box)
                self.canvas.current_boxes.append(box)

    def submit_labels(self):
        if not self.current_doc:
            self.statusBar().showMessage("No image selected to submit!", 3000)
            return
            
        labels = self.canvas.get_yolo_labels()
        if not labels and self.current_mode == "pending":
            msg = QMessageBox.question(self, "No Labels", 
                                     "You haven't drawn any boxes. Submit as 'No Dust'?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if msg == QMessageBox.StandardButton.No:
                return

        try:
            print(f"[INFO] Submitting {len(labels)} labels for {self.current_doc}...")
            self.supabase.table(SUPABASE_TABLE).update({
                "status": "labeled",
                "labels": labels,
                "labeled_at": datetime.now().isoformat()
            }).eq("filename", self.current_doc).execute()
            
            self.statusBar().showMessage(f"Successfully submitted: {self.current_doc}", 3000)
            
            # Move to next
            next_row = self.img_list.currentRow() + 1
            if next_row < self.img_list.count():
                self.img_list.setCurrentRow(next_row)
                self.on_item_clicked(self.img_list.item(next_row))
            else:
                self.load_data(self.current_mode)
                
        except Exception as e:
            print(f"[ERROR] Failed to submit: {e}")
            QMessageBox.critical(self, "Submission Failed", f"Firestore Error: {e}")

    def delete_image(self):
        if not self.current_doc: return
        msg = QMessageBox.question(self, "Delete", "Delete this image from cloud?", 
                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if msg == QMessageBox.StandardButton.Yes:
            item = self.img_list.currentItem()
            storage_path = item.data(Qt.ItemDataRole.UserRole + 1)
            
            # Delete from storage & firestore
            self.bucket.blob(storage_path).delete()
            self.db.collection("images").document(self.current_doc).delete()
            
            self.statusBar().showMessage("Image Deleted", 3000)
            self.load_data(self.current_mode)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"[ERROR] Could not connect to Supabase: {e}")
        sys.exit(1)
    
    login = LoginDialog(supabase)
    main_window = RyucentStudio()
    
    def on_success():
        main_window.show()
        
    login.authenticated.connect(on_success)
    login.show()
    
    sys.exit(app.exec())
