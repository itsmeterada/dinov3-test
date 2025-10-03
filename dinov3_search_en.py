import sys
import os
import sqlite3
import numpy as np
from PIL import Image
import hashlib
import io
import threading
import queue
from pathlib import Path
import glob
import time
import zipfile
import tempfile
from logging import getLogger

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QComboBox,
                             QLineEdit, QTextEdit, QFrame, QScrollArea,
                             QGridLayout, QFileDialog, QMessageBox, QProgressBar,
                             QSplitter, QGroupBox, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QTimer, QMimeData,
                          QPropertyAnimation, QEasingCurve, QRect, QSize)
from PyQt6.QtGui import (QPixmap, QFont, QPalette, QColor, QDragEnterEvent,
                         QDropEvent, QIcon, QPainter, QBrush, QLinearGradient)

import torch
from transformers import AutoModel, AutoImageProcessor
import torchvision.transforms as transforms

logger = getLogger(__name__)

# ======================
# DINOv3 Parameters
# ======================

# DINOv3 model name (from Hugging Face)
MODEL_NAME = "facebook/dinov2-base"
IMAGE_SIZE = 224

# Color palette (matching DINOv3 brand colors)
COLORS = {
    'primary': '#4A90E2',      # DINOv3 Blue
    'secondary': '#7AB8F5',    # Light Blue
    'background': '#FFFFFF',   # White
    'card': '#F8F9FA',         # Light Gray
    'text': '#2C3E50',         # Dark Gray
    'border': '#E9ECEF',       # Border Gray
    'success': '#28A745',      # Success Color
    'warning': '#FFC107',      # Warning Color
    'danger': '#DC3545'        # Error Color
}

def fetch_base_path() -> str:
    """Function to get the base path"""
    # Check if running as PyInstaller executable
    if getattr(sys, "frozen", False):
        # Get the path of the EXE executable
        return os.path.dirname(sys.argv[0])
    else:
        # Get the path of the script file
        return os.path.dirname(os.path.abspath(__file__))

mypath = fetch_base_path()

class ModelLoaderThread(QThread):
    """Worker thread for model loading"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, object, object)

    def run(self):
        try:
            self.progress.emit("Downloading DINOv3 model...")

            # Device configuration
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if device.type == "cuda":
                self.progress.emit("Loading model in GPU mode...")
            else:
                self.progress.emit("Loading model in CPU mode...")

            # Load DINOv3 model and processor
            self.progress.emit("Loading DINOv3 model...")
            model = AutoModel.from_pretrained(MODEL_NAME)
            processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

            model = model.to(device)
            model.eval()

            self.progress.emit("DINOv3 model loading completed")
            self.finished.emit(True, model, processor)

        except Exception as e:
            self.progress.emit(f"Error: {e}")
            self.finished.emit(False, None, None)

class DirectoryProcessorThread(QThread):
    """Worker thread for directory processing"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    count_updated = pyqtSignal()

    def __init__(self, directory_path, db_path, feature_extractor):
        super().__init__()
        self.directory_path = directory_path
        self.db_path = db_path
        self.feature_extractor = feature_extractor

    def run(self):
        try:
            self.progress.emit(f"Processing folder: {self.directory_path}")

            # Search for image files
            image_files = []
            for root, _, files in os.walk(self.directory_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                        image_files.append(os.path.join(root, file))

            total_files = len(image_files)
            processed_files = 0
            added_files = 0
            skipped_files = 0

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for image_path in image_files:
                try:
                    file_hash = self.calculate_file_hash(image_path)

                    # Check if already exists
                    cursor.execute("SELECT id FROM images WHERE file_hash = ?", (file_hash,))
                    if cursor.fetchone():
                        skipped_files += 1
                        processed_files += 1
                        continue

                    # Add new image only
                    if self.add_image_to_database(image_path, cursor):
                        added_files += 1

                    processed_files += 1
                    if processed_files % 5 == 0 or processed_files == total_files:
                        self.progress.emit(
                            f"Processing: {processed_files}/{total_files} "
                            f"(Added: {added_files}, Skipped: {skipped_files})"
                        )

                except Exception as e:
                    self.progress.emit(f"Error ({os.path.basename(image_path)}): {e}")

            conn.commit()
            conn.close()

            self.finished.emit(
                f"Folder processing completed. {processed_files} files processed, "
                f"{added_files} files added, {skipped_files} files skipped."
            )
            self.count_updated.emit()

        except Exception as e:
            self.finished.emit(f"Directory processing error: {e}")

    def calculate_file_hash(self, file_path):
        """Calculate file hash"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def add_image_to_database(self, file_path, cursor):
        """Add image to database"""
        try:
            image = Image.open(file_path)
            file_hash = self.calculate_file_hash(file_path)

            # Create thumbnail
            thumbnail = image.copy()
            thumbnail.thumbnail((150, 150))
            buffer = io.BytesIO()
            thumbnail.save(buffer, format="JPEG")
            thumbnail_bytes = buffer.getvalue()

            # Insert image information
            cursor.execute(
                "INSERT INTO images (file_path, file_hash, thumbnail) VALUES (?, ?, ?)",
                (file_path, file_hash, thumbnail_bytes)
            )
            image_id = cursor.lastrowid

            # Extract features
            feature_vector = self.feature_extractor.extract_features(image)
            if feature_vector is None:
                return False

            # Insert features
            cursor.execute(
                "INSERT INTO features (image_id, feature_vector) VALUES (?, ?)",
                (image_id, feature_vector.tobytes())
            )

            return True
        except Exception as e:
            print(f"Error ({os.path.basename(file_path)}): {e}")
            return False

class ZipProcessorThread(QThread):
    """Worker thread for ZIP processing"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    count_updated = pyqtSignal()

    def __init__(self, zip_path, db_path, feature_extractor):
        super().__init__()
        self.zip_path = zip_path
        self.db_path = db_path
        self.feature_extractor = feature_extractor

    def run(self):
        temp_dir = None
        try:
            self.progress.emit(f"Processing ZIP file: {os.path.basename(self.zip_path)}")

            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="dinov3_zip_")

            # Extract ZIP file
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # Get list of files in ZIP
                file_list = zip_ref.namelist()

                # Filter only image files
                image_files = [f for f in file_list
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))
                              and not f.startswith('__MACOSX/')  # Exclude macOS hidden files
                              and not os.path.basename(f).startswith('.')]  # Exclude hidden files

                if not image_files:
                    self.finished.emit("No valid image files found in ZIP file.")
                    return

                total_files = len(image_files)
                processed_files = 0
                added_files = 0
                skipped_files = 0

                self.progress.emit(f"Image files in ZIP: {total_files}")

                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                for file_path in image_files:
                    try:
                        # Extract file to temporary directory
                        extracted_path = zip_ref.extract(file_path, temp_dir)

                        # Calculate file hash
                        file_hash = self.calculate_file_hash(extracted_path)

                        # Check if already exists
                        cursor.execute("SELECT id FROM images WHERE file_hash = ?", (file_hash,))
                        if cursor.fetchone():
                            skipped_files += 1
                            processed_files += 1
                            continue

                        # Use path inside ZIP as recorded file path
                        zip_file_path = f"{os.path.basename(self.zip_path)}:{file_path}"

                        # Add new image only
                        if self.add_image_to_database(extracted_path, zip_file_path, file_hash, cursor):
                            added_files += 1

                        processed_files += 1
                        if processed_files % 5 == 0 or processed_files == total_files:
                            self.progress.emit(
                                f"Processing: {processed_files}/{total_files} "
                                f"(Added: {added_files}, Skipped: {skipped_files})"
                            )

                    except Exception as e:
                        self.progress.emit(f"Error ({os.path.basename(file_path)}): {e}")
                        processed_files += 1

                conn.commit()
                conn.close()

                self.finished.emit(
                    f"ZIP file processing completed. {processed_files} files processed, "
                    f"{added_files} files added, {skipped_files} files skipped."
                )
                self.count_updated.emit()

        except Exception as e:
            self.finished.emit(f"ZIP file processing error: {e}")
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Failed to delete temporary directory: {e}")

    def calculate_file_hash(self, file_path):
        """Calculate file hash"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def add_image_to_database(self, file_path, zip_file_path, file_hash, cursor):
        """Add image to database"""
        try:
            image = Image.open(file_path)

            # Create thumbnail
            thumbnail = image.copy()
            thumbnail.thumbnail((150, 150))
            buffer = io.BytesIO()
            thumbnail.save(buffer, format="JPEG")
            thumbnail_bytes = buffer.getvalue()

            # Insert image information (record path inside ZIP)
            cursor.execute(
                "INSERT INTO images (file_path, file_hash, thumbnail) VALUES (?, ?, ?)",
                (zip_file_path, file_hash, thumbnail_bytes)
            )
            image_id = cursor.lastrowid

            # Extract features
            feature_vector = self.feature_extractor.extract_features(image)
            if feature_vector is None:
                return False

            # Insert features
            cursor.execute(
                "INSERT INTO features (image_id, feature_vector) VALUES (?, ?)",
                (image_id, feature_vector.tobytes())
            )

            return True
        except Exception as e:
            print(f"Error ({os.path.basename(file_path)}): {e}")
            return False

class FeatureExtractor:
    """DINOv3 feature extraction class"""
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_features(self, image):
        """Extract features from image"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token features
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Normalize
            norm = np.linalg.norm(features, axis=1, keepdims=True)
            if norm > 0:
                features = features / norm

            return features.squeeze()
        except Exception as e:
            print(f"Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None

class SimilarImageWidget(QWidget):
    """Similar image display widget"""
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Image display label
        self.image_label = QLabel()
        self.image_label.setFixedSize(150, 150)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                background-color: {COLORS['card']};
            }}
        """)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        # Information display label
        self.info_label = QLabel()
        self.info_label.setFixedHeight(60)
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet(f"""
            QLabel {{
                background-color: white;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px;
                font-size: 10px;
                font-weight: bold;
            }}
        """)
        layout.addWidget(self.info_label)

    def set_image_data(self, image_data):
        """Set image data"""
        if not image_data:
            self.clear()
            return

        # Display image (preserve aspect ratio)
        pixmap = QPixmap()
        pixmap.loadFromData(image_data["thumbnail"])

        # Scale while preserving aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

        # Display information
        similarity_percentage = image_data["similarity"] * 100
        file_path = image_data["file_path"]

        # Determine if image is from ZIP and display filename
        if ':' in file_path:
            # Image from ZIP file
            zip_name, internal_path = file_path.split(':', 1)
            file_name = os.path.basename(internal_path)
            if len(file_name) > 12:
                file_name = file_name[:10] + "..."
            display_name = f"📦{file_name}"
        else:
            # Regular file
            file_name = os.path.basename(file_path)
            if len(file_name) > 12:
                file_name = file_name[:10] + "..."
            display_name = file_name

        info_text = f"{display_name}\nSimilarity: {similarity_percentage:.1f}%"

        # Color-code by similarity
        if similarity_percentage >= 80:
            color = COLORS['success']
            bg_color = "#D4F6D4"
        elif similarity_percentage >= 60:
            color = COLORS['warning']
            bg_color = "#FFF3CD"
        else:
            color = COLORS['danger']
            bg_color = "#F8D7DA"

        self.info_label.setText(info_text)
        self.info_label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                border: 1px solid {color};
                border-radius: 4px;
                padding: 4px;
                font-size: 10px;
                font-weight: bold;
                color: {color};
            }}
        """)

    def clear(self):
        """Clear display"""
        self.image_label.clear()
        self.image_label.setText("No Image")
        self.info_label.clear()
        self.info_label.setStyleSheet(f"""
            QLabel {{
                background-color: white;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px;
                font-size: 10px;
                font-weight: bold;
            }}
        """)

class DropAreaWidget(QLabel):
    """Drag & Drop area"""
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.current_pixmap = None
        self.is_image_displayed = False
        self.setup_ui()

    def setup_ui(self):
        """Initial UI setup"""
        if not self.is_image_displayed:
            self.setMinimumHeight(200)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setText("Drag and drop image, folder, or ZIP file here")
            self.setPixmap(QPixmap())
            self.setStyleSheet(f"""
                QLabel {{
                    border: 3px dashed {COLORS['border']};
                    border-radius: 12px;
                    background-color: {COLORS['card']};
                    color: {COLORS['text']};
                    font-size: 16px;
                    font-weight: bold;
                }}
            """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(f"""
                QLabel {{
                    border: 3px dashed {COLORS['primary']};
                    border-radius: 12px;
                    background-color: {COLORS['secondary']}20;
                    color: {COLORS['primary']};
                    font-size: 16px;
                    font-weight: bold;
                }}
            """)

    def dragLeaveEvent(self, event):
        if self.is_image_displayed:
            self.restore_image_display_style()
        else:
            self.setup_ui()

    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            files.append(url.toLocalFile())

        if files:
            self.files_dropped.emit(files)

        if self.is_image_displayed:
            self.restore_image_display_style()
        else:
            self.setup_ui()

    def restore_image_display_style(self):
        """Restore image display style"""
        self.setStyleSheet(f"""
            QLabel {{
                border: 2px solid {COLORS['primary']};
                border-radius: 12px;
                background-color: white;
                padding: 10px;
            }}
        """)

    def set_pixmap_from_pil(self, pil_image):
        """Display from PIL image"""
        try:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Convert PIL image to byte array and load with QPixmap
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            pixmap = QPixmap()
            if pixmap.loadFromData(img_buffer.getvalue()):
                self.display_pixmap(pixmap)
            else:
                self.show_error_message("Failed to convert image")

        except Exception as e:
            print(f"PIL image display error: {e}")
            import traceback
            traceback.print_exc()
            self.show_error_message("Failed to display image")

    def display_pixmap(self, pixmap):
        """Display QPixmap"""
        try:
            # Resize with fixed size while preserving aspect ratio
            target_size = QSize(400, 300)
            scaled_pixmap = pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.current_pixmap = scaled_pixmap
            self.is_image_displayed = True

            self.setPixmap(scaled_pixmap)
            self.setText("")
            self.restore_image_display_style()

        except Exception as e:
            print(f"QPixmap display error: {e}")
            self.show_error_message("Failed to display image")

    def show_error_message(self, message):
        """Display error message"""
        self.setText(message)
        self.setStyleSheet(f"""
            QLabel {{
                border: 2px solid {COLORS['danger']};
                border-radius: 12px;
                background-color: #FFF5F5;
                color: {COLORS['danger']};
                font-size: 16px;
                font-weight: bold;
            }}
        """)
        QTimer.singleShot(3000, self.restore_previous_state)

    def restore_previous_state(self):
        """Restore to previous state"""
        if self.is_image_displayed and self.current_pixmap:
            self.setPixmap(self.current_pixmap)
            self.setText("")
            self.restore_image_display_style()
        else:
            self.setup_ui()

    def clear(self):
        """Clear display completely"""
        self.current_pixmap = None
        self.is_image_displayed = False
        QLabel.clear(self)
        self.setText("Drag and drop image, folder, or ZIP file here")
        self.setup_ui()

class DINOv3ImageSearchApp(QMainWindow):
    """Main application class"""

    def __init__(self):
        super().__init__()
        self.feature_extractor = None
        self.model_loaded = False
        self.current_image = None
        self.current_image_path = None
        self.init_database()
        self.setup_ui()
        self.setup_styles()
        self.load_models()

    def init_database(self):
        """Initialize database"""
        self.db_files = []
        self.scan_db_files()

    def scan_db_files(self):
        """Scan database files"""
        self.db_files = []
        for db_file in glob.glob(os.path.join(mypath, "*.db")):
            self.db_files.append(os.path.basename(db_file))

        # Create default database file if it doesn't exist
        default_db_file = "image_features_dinov3.db"
        if default_db_file not in self.db_files:
            default_db_path = os.path.join(mypath, default_db_file)
            self.initialize_single_database(default_db_path)
            self.db_files.append(default_db_file)

    def initialize_single_database(self, db_path):
        """Initialize single database"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            file_hash TEXT UNIQUE,
            thumbnail BLOB
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            image_id INTEGER,
            feature_vector BLOB,
            FOREIGN KEY (image_id) REFERENCES images (id),
            PRIMARY KEY (image_id)
        )
        ''')

        conn.commit()
        conn.close()

    def setup_ui(self):
        """UI setup"""
        self.setWindowTitle("DINOv3 Image Search Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Main layout
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header section
        self.setup_header(main_layout)

        # Control section
        self.setup_controls(main_layout)

        # Content section (splitter)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        # Left side: Image display area
        self.setup_image_area(splitter)

        # Right side: Search results area
        self.setup_results_area(splitter)

        # Status bar
        self.setup_status_bar()

    def setup_header(self, layout):
        """Header section setup"""
        header_frame = QFrame()
        header_frame.setFixedHeight(80)
        header_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['primary']}, stop:1 {COLORS['secondary']});
                border-radius: 12px;
                margin-bottom: 10px;
            }}
        """)

        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 10, 20, 10)

        # DINOv3 logo
        logo_label = QLabel("DINOv3")
        logo_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 28px;
                font-weight: bold;
                font-family: 'Arial', sans-serif;
            }
        """)
        header_layout.addWidget(logo_label)

        header_layout.addStretch()

        # Title
        title_label = QLabel("Image Search Tool")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 20px;
                font-weight: bold;
            }
        """)
        header_layout.addWidget(title_label)

        layout.addWidget(header_frame)

    def setup_controls(self, layout):
        """Control section setup"""
        controls_frame = QFrame()
        controls_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['card']};
                border-radius: 8px;
                padding: 10px;
            }}
        """)

        controls_layout = QVBoxLayout(controls_frame)

        # Database management row
        db_layout = QHBoxLayout()

        db_layout.addWidget(QLabel("Database:"))

        self.db_combo = QComboBox()
        self.db_combo.addItems(self.db_files)
        if self.db_files:
            self.db_combo.setCurrentText(self.db_files[0])
        self.db_combo.currentTextChanged.connect(self.on_database_change)
        db_layout.addWidget(self.db_combo)

        self.image_count_label = QLabel("(0 images)")
        db_layout.addWidget(self.image_count_label)

        db_layout.addStretch()

        # Database operation buttons
        self.new_db_entry = QLineEdit()
        self.new_db_entry.setPlaceholderText("New DB name")
        self.new_db_entry.setMaximumWidth(150)
        db_layout.addWidget(self.new_db_entry)

        create_db_btn = QPushButton("Create")
        create_db_btn.clicked.connect(self.create_new_database)
        db_layout.addWidget(create_db_btn)

        clear_db_btn = QPushButton("Clear DB")
        clear_db_btn.clicked.connect(self.clear_database)
        db_layout.addWidget(clear_db_btn)

        delete_db_btn = QPushButton("Delete DB")
        delete_db_btn.clicked.connect(self.delete_database)
        db_layout.addWidget(delete_db_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_db_list)
        db_layout.addWidget(refresh_btn)

        controls_layout.addLayout(db_layout)

        # File operation row
        file_layout = QHBoxLayout()

        select_image_btn = QPushButton("Select Image")
        select_image_btn.clicked.connect(self.select_image)
        file_layout.addWidget(select_image_btn)

        select_folder_btn = QPushButton("Select Folder")
        select_folder_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(select_folder_btn)

        select_zip_btn = QPushButton("Select ZIP")
        select_zip_btn.clicked.connect(self.select_zip)
        file_layout.addWidget(select_zip_btn)

        clear_image_btn = QPushButton("Clear Image")
        clear_image_btn.clicked.connect(self.clear_image_display)
        file_layout.addWidget(clear_image_btn)

        self.add_to_db_btn = QPushButton("Add to DB")
        self.add_to_db_btn.clicked.connect(self.add_current_image_to_db)
        self.add_to_db_btn.setEnabled(False)
        self.add_to_db_btn.setStyleSheet(f"""
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: #999999;
            }}
        """)
        file_layout.addWidget(self.add_to_db_btn)

        file_layout.addStretch()

        controls_layout.addLayout(file_layout)

        layout.addWidget(controls_frame)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid {COLORS['border']};
                border-radius: 5px;
                text-align: center;
                background-color: {COLORS['card']};
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)

    def setup_image_area(self, splitter):
        """Image display area setup"""
        image_frame = QFrame()
        image_frame.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)

        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title_label = QLabel("Image Display Area")
        title_label.setStyleSheet(f"""
            QLabel {{
                font-size: 16px;
                font-weight: bold;
                color: {COLORS['text']};
                padding: 5px;
            }}
        """)
        image_layout.addWidget(title_label)

        # Drop area
        self.drop_area = DropAreaWidget()
        self.drop_area.files_dropped.connect(self.handle_dropped_files)
        image_layout.addWidget(self.drop_area, 1)

        splitter.addWidget(image_frame)

    def setup_results_area(self, splitter):
        """Search results area setup"""
        results_frame = QFrame()
        results_frame.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)

        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title_label = QLabel("Search Results")
        title_label.setStyleSheet(f"""
            QLabel {{
                font-size: 16px;
                font-weight: bold;
                color: {COLORS['text']};
                padding: 5px;
            }}
        """)
        results_layout.addWidget(title_label)

        # Similar image display area (2 rows x 5 columns grid)
        similar_frame = QFrame()
        similar_layout = QGridLayout(similar_frame)
        similar_layout.setContentsMargins(5, 5, 5, 5)
        similar_layout.setSpacing(10)

        # Create 10 similar image widgets
        self.similar_widgets = []
        for row in range(2):
            for col in range(5):
                widget = SimilarImageWidget()
                self.similar_widgets.append(widget)
                similar_layout.addWidget(widget, row, col)

        results_layout.addWidget(similar_frame)
        results_layout.addStretch()

        splitter.addWidget(results_frame)
        splitter.setSizes([600, 400])

    def setup_status_bar(self):
        """Status bar setup"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Preparing...")
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['card']};
                border-top: 1px solid {COLORS['border']};
                color: {COLORS['text']};
            }}
        """)

    def setup_styles(self):
        """Overall style setup"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
            }}

            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                min-width: 80px;
            }}

            QPushButton:hover {{
                background-color: {COLORS['secondary']};
            }}

            QPushButton:pressed {{
                background-color: #3A7BC8;
            }}

            QComboBox {{
                padding: 5px;
                border: 2px solid {COLORS['border']};
                border-radius: 4px;
                background-color: white;
                min-width: 150px;
            }}

            QComboBox:focus {{
                border-color: {COLORS['primary']};
            }}

            QLineEdit {{
                padding: 8px;
                border: 2px solid {COLORS['border']};
                border-radius: 4px;
                background-color: white;
                font-size: 14px;
            }}

            QLineEdit:focus {{
                border-color: {COLORS['primary']};
            }}

            QLabel {{
                color: {COLORS['text']};
            }}
        """)

    def load_models(self):
        """Load models"""
        self.model_thread = ModelLoaderThread()
        self.model_thread.progress.connect(self.update_status)
        self.model_thread.finished.connect(self.on_model_loaded)
        self.model_thread.start()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

    def on_model_loaded(self, success, model, processor):
        """Processing on model load completion"""
        self.progress_bar.setVisible(False)

        if success:
            self.feature_extractor = FeatureExtractor(model, processor)
            self.model_loaded = True
            self.update_status("DINOv3 model loading completed.")
            self.update_image_count()
        else:
            self.update_status("Failed to load DINOv3 model.")
            QMessageBox.critical(self, "Error", "Failed to load DINOv3 model.")

    def update_status(self, message):
        """Update status message"""
        self.status_bar.showMessage(message)

    def update_image_count(self):
        """Update image count"""
        if not self.db_combo.currentText():
            self.image_count_label.setText("(0 images)")
            return

        try:
            db_path = os.path.join(mypath, self.db_combo.currentText())
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
            if cursor.fetchone() is None:
                self.initialize_single_database(db_path)
                count = 0
            else:
                cursor.execute("SELECT COUNT(*) FROM images")
                count = cursor.fetchone()[0]

            conn.close()
            self.image_count_label.setText(f"({count} images)")
        except Exception as e:
            self.image_count_label.setText("(Error)")
            print(f"Image count retrieval error: {e}")

    def on_database_change(self, db_name):
        """Processing on database change"""
        self.update_image_count()
        for widget in self.similar_widgets:
            widget.clear()

    def handle_dropped_files(self, files):
        """Handle dropped files"""
        if not self.model_loaded:
            QMessageBox.information(self, "Information", "DINOv3 model is not yet loaded. Please wait.")
            return

        for file_path in files:
            if os.path.isfile(file_path):
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    self.process_image_file(file_path)
                    break
                elif file_path.lower().endswith('.zip'):
                    self.process_zip_file(file_path)
                    break
            elif os.path.isdir(file_path):
                self.process_directory(file_path)
                break

    def select_image(self):
        """Select image file"""
        if not self.model_loaded:
            QMessageBox.information(self, "Information", "DINOv3 model is not yet loaded.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.webp)"
        )

        if file_path:
            self.process_image_file(file_path)

    def select_folder(self):
        """Select folder"""
        if not self.model_loaded:
            QMessageBox.information(self, "Information", "DINOv3 model is not yet loaded.")
            return

        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")

        if folder_path:
            self.process_directory(folder_path)

    def select_zip(self):
        """Select ZIP file"""
        if not self.model_loaded:
            QMessageBox.information(self, "Information", "DINOv3 model is not yet loaded.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select ZIP File", "",
            "ZIP Files (*.zip)"
        )

        if file_path:
            self.process_zip_file(file_path)

    def process_image_file(self, file_path):
        """Process single image file"""
        try:
            image = Image.open(file_path)

            self.current_image = image.copy()
            self.current_image_path = file_path

            self.add_to_db_btn.setEnabled(True)
            self.drop_area.set_pixmap_from_pil(image)

            QApplication.processEvents()

            self.extract_and_search_features(image)

        except Exception as e:
            print(f"Image processing error: {e}")
            import traceback
            traceback.print_exc()
            self.update_status(f"Image processing error: {e}")
            QMessageBox.warning(self, "Warning", f"An error occurred during image processing:\n{str(e)[:100]}...")

    def process_zip_file(self, zip_path):
        """Process ZIP file"""
        if not self.feature_extractor:
            QMessageBox.information(self, "Information", "Model is not yet loaded.")
            return

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                pass
        except zipfile.BadZipFile:
            QMessageBox.warning(self, "Error", "Invalid ZIP file.")
            return
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load ZIP file: {e}")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        db_path = os.path.join(mypath, self.db_combo.currentText())

        self.zip_thread = ZipProcessorThread(
            zip_path, db_path, self.feature_extractor
        )
        self.zip_thread.progress.connect(self.update_status)
        self.zip_thread.finished.connect(self.on_zip_processed)
        self.zip_thread.count_updated.connect(self.update_image_count)
        self.zip_thread.start()

    def on_zip_processed(self, message):
        """Processing on ZIP processing completion"""
        self.progress_bar.setVisible(False)
        self.update_status(message)

    def extract_and_search_features(self, image):
        """Execute feature extraction and search"""
        try:
            self.update_status("Extracting features...")

            feature_vector = self.feature_extractor.extract_features(image)

            if feature_vector is None:
                self.update_status("Failed to extract features.")
                QMessageBox.warning(self, "Warning", "Failed to extract features.")
                return

            self.search_similar_images(feature_vector)

        except Exception as e:
            print(f"Feature extraction error: {e}")
            self.update_status(f"Feature extraction error: {e}")
            QMessageBox.warning(self, "Warning", f"An error occurred during feature extraction:\n{str(e)[:100]}...")

    def search_similar_images(self, query_features):
        """Search for similar images"""
        try:
            db_path = os.path.join(mypath, self.db_combo.currentText())
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT image_id, feature_vector FROM features")
            results = cursor.fetchall()

            if not results:
                self.update_status("No images registered in database.")
                for widget in self.similar_widgets:
                    widget.clear()
                conn.close()
                return

            similarities = []
            for result_id, result_bytes in results:
                result_features = np.frombuffer(result_bytes, dtype=np.float32)
                similarity = np.dot(query_features, result_features) / (
                    np.linalg.norm(query_features) * np.linalg.norm(result_features)
                )
                similarities.append((result_id, similarity))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_10 = similarities[:10]

            similar_images = []
            for similar_id, similarity in top_10:
                cursor.execute("SELECT file_path, thumbnail FROM images WHERE id = ?", (similar_id,))
                result = cursor.fetchone()
                if result:
                    file_path, thumbnail_bytes = result
                    similar_images.append({
                        "id": similar_id,
                        "file_path": file_path,
                        "thumbnail": thumbnail_bytes,
                        "similarity": similarity
                    })

            conn.close()

            self.display_similar_images(similar_images)
            self.update_status(f"Similar image search completed. Displaying top {len(similar_images)}.")

        except Exception as e:
            self.update_status(f"Similar image search error: {e}")
            QMessageBox.warning(self, "Warning", f"An error occurred during similar image search:\n{str(e)[:100]}...")

    def display_similar_images(self, similar_images):
        """Display similar images"""
        for i, widget in enumerate(self.similar_widgets):
            if i < len(similar_images):
                widget.set_image_data(similar_images[i])
            else:
                widget.clear()

    def process_directory(self, directory_path):
        """Process directory"""
        if not self.feature_extractor:
            QMessageBox.information(self, "Information", "Model is not yet loaded.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        db_path = os.path.join(mypath, self.db_combo.currentText())

        self.dir_thread = DirectoryProcessorThread(
            directory_path, db_path, self.feature_extractor
        )
        self.dir_thread.progress.connect(self.update_status)
        self.dir_thread.finished.connect(self.on_directory_processed)
        self.dir_thread.count_updated.connect(self.update_image_count)
        self.dir_thread.start()

    def on_directory_processed(self, message):
        """Processing on directory processing completion"""
        self.progress_bar.setVisible(False)
        self.update_status(message)

    def create_new_database(self):
        """Create new database"""
        db_name = self.new_db_entry.text().strip()
        if not db_name:
            QMessageBox.warning(self, "Warning", "Please enter a database name.")
            return

        if not db_name.lower().endswith('.db'):
            db_name += '.db'

        db_path = os.path.join(mypath, db_name)
        if os.path.exists(db_path):
            QMessageBox.warning(self, "Warning", f"Database '{db_name}' already exists.")
            return

        self.initialize_single_database(db_path)
        self.refresh_db_list()
        self.db_combo.setCurrentText(db_name)
        self.update_image_count()

        QMessageBox.information(self, "Information", f"Created database '{db_name}'.")

    def clear_database(self):
        """Clear database"""
        if not self.db_combo.currentText():
            QMessageBox.warning(self, "Warning", "No database selected.")
            return

        reply = QMessageBox.question(
            self, "Confirm",
            f"Clear contents of database '{self.db_combo.currentText()}'. "
            "This operation cannot be undone. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                db_path = os.path.join(mypath, self.db_combo.currentText())
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                cursor.execute("DELETE FROM features")
                cursor.execute("DELETE FROM images")
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='images'")

                conn.commit()
                conn.close()

                QMessageBox.information(self, "Information", f"Cleared database '{self.db_combo.currentText()}'.")
                self.update_image_count()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while clearing database: {e}")

    def delete_database(self):
        """Delete database file"""
        if not self.db_combo.currentText():
            QMessageBox.warning(self, "Warning", "No database selected.")
            return

        current_db = self.db_combo.currentText()

        reply = QMessageBox.question(
            self, "Confirm",
            f"Permanently delete database file '{current_db}'. "
            "This operation cannot be undone. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                db_path = os.path.join(mypath, current_db)
                os.remove(db_path)
                self.refresh_db_list()

                QMessageBox.information(self, "Information", f"Deleted database file '{current_db}'.")

                if self.db_files:
                    self.db_combo.setCurrentText(self.db_files[0])
                    self.update_image_count()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while deleting database: {e}")

    def clear_image_display(self):
        """Clear image display area"""
        self.drop_area.clear()
        self.current_image = None
        self.current_image_path = None
        self.add_to_db_btn.setEnabled(False)

        for widget in self.similar_widgets:
            widget.clear()
        self.update_status("Cleared image display.")

    def add_current_image_to_db(self):
        """Add currently displayed image to database"""
        if not self.current_image or not self.model_loaded:
            QMessageBox.warning(self, "Warning", "No image to add or model not loaded.")
            return

        if not self.db_combo.currentText():
            QMessageBox.warning(self, "Warning", "No database selected.")
            return

        try:
            if self.current_image_path:
                file_hash = self.calculate_file_hash(self.current_image_path)
                file_path = self.current_image_path
            else:
                file_path = f"dropped_image_{int(time.time())}.png"
                img_buffer = io.BytesIO()
                self.current_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                file_hash = hashlib.md5(img_buffer.getvalue()).hexdigest()

            db_path = os.path.join(mypath, self.db_combo.currentText())
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM images WHERE file_hash = ?", (file_hash,))
            existing = cursor.fetchone()

            if existing:
                QMessageBox.information(self, "Information", "This image is already registered in the database.")
                conn.close()
                return

            thumbnail = self.current_image.copy()
            thumbnail.thumbnail((150, 150))
            buffer = io.BytesIO()
            thumbnail.save(buffer, format="JPEG")
            thumbnail_bytes = buffer.getvalue()

            cursor.execute(
                "INSERT INTO images (file_path, file_hash, thumbnail) VALUES (?, ?, ?)",
                (file_path, file_hash, thumbnail_bytes)
            )
            image_id = cursor.lastrowid

            self.update_status("Extracting features...")
            feature_vector = self.feature_extractor.extract_features(self.current_image)

            if feature_vector is None:
                cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
                conn.commit()
                conn.close()
                QMessageBox.warning(self, "Warning", "Failed to extract features. Could not add image.")
                return

            cursor.execute(
                "INSERT INTO features (image_id, feature_vector) VALUES (?, ?)",
                (image_id, feature_vector.tobytes())
            )

            conn.commit()
            conn.close()

            file_name = os.path.basename(file_path) if file_path else "dropped_image"
            QMessageBox.information(self, "Success", f"Added image \"{file_name}\" to database.")

            self.update_image_count()
            self.update_status(f"Added image \"{file_name}\" to database.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while adding to database:\n{str(e)}")
            print(f"DB add error: {e}")
            import traceback
            traceback.print_exc()

    def calculate_file_hash(self, file_path):
        """Calculate file hash"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def refresh_db_list(self):
        """Refresh database list"""
        self.scan_db_files()
        self.db_combo.clear()
        self.db_combo.addItems(self.db_files)
        self.update_image_count()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    app.setApplicationName("DINOv3 Image Search")
    app.setApplicationVersion("1.0")

    window = DINOv3ImageSearchApp()
    window.show()

    os.chdir(mypath)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
