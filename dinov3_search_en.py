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

# DINOv3モデルのインポートのためにパスを追加
dinov3_cache_path = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "facebookresearch_dinov3_main")
if os.path.exists(dinov3_cache_path) and dinov3_cache_path not in sys.path:
    sys.path.insert(0, dinov3_cache_path)

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

# モデル設定
USE_DINOV3 = True  # DINOv3を使用

# 利用可能なDINOv3モデルリスト
AVAILABLE_MODELS = {
    "ViT-S/16 (21M)": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "ViT-S+/16 (29M)": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "ViT-B/16 (86M)": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "ViT-L/16 (300M)": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "ViT-H+/16 (840M)": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "ViT-7B/16 (6.7B)": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "ConvNeXt Tiny (29M)": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "ConvNeXt Small (50M)": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "ConvNeXt Base (89M)": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "ConvNeXt Large (198M)": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    "ViT-L/16 SAT (300M)": "facebook/dinov3-vitl16-pretrain-sat493m",
    "ViT-7B/16 SAT (6.7B)": "facebook/dinov3-vit7b16-pretrain-sat493m",
}

# デフォルトモデル
DEFAULT_MODEL = "ViT-B/16 (86M)"

IMAGE_SIZE = 224

# カラーパレット（DINOv3のブランドカラーに合わせて）
COLORS = {
    'primary': '#4A90E2',      # DINOv3 ブルー
    'secondary': '#7AB8F5',    # 薄いブルー
    'background': '#FFFFFF',   # 白
    'card': '#F8F9FA',         # ライトグレー
    'text': '#2C3E50',         # ダークグレー
    'border': '#E9ECEF',       # ボーダーグレー
    'success': '#28A745',      # 成功色
    'warning': '#FFC107',      # 警告色
    'danger': '#DC3545'        # エラー色
}

def fetch_base_path() -> str:
    """基準パスを取得する関数"""
    # PyInstallerで実行されているかどうかをチェック
    if getattr(sys, "frozen", False):
        # EXEの実行ファイルのパスを取得
        return os.path.dirname(sys.argv[0])
    else:
        # スクリプトの実行ファイルのパスを取得
        return os.path.dirname(os.path.abspath(__file__))

mypath = fetch_base_path()

class ModelLoaderThread(QThread):
    """モデル読み込み用のワーカースレッド"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, object, object)

    def __init__(self, model_name=None):
        super().__init__()
        self.model_name = model_name or AVAILABLE_MODELS[DEFAULT_MODEL]

    def run(self):
        try:
            self.progress.emit("DINOv3モデルをダウンロード中...")

            # デバイスの設定
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if device.type == "cuda":
                self.progress.emit("GPUモードでモデルを読み込み中...")
            else:
                self.progress.emit("CPUモードでモデルを読み込み中...")

            # モデルの読み込み
            self.progress.emit("DINOv3モデルを読み込み中...")

            # DINOv3モデルをHugging Faceからロード
            try:
                from transformers import AutoImageProcessor, AutoModel

                self.progress.emit(f"Downloading {self.model_name} from Hugging Face...")
                self.progress.emit("(First time may take several minutes)")

                processor = AutoImageProcessor.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)

                self.progress.emit("✓ DINOv3 model loaded successfully")

            except Exception as load_error:
                self.progress.emit(f"エラー: {str(load_error)[:200]}")
                raise

            model = model.to(device)
            model.eval()

            self.progress.emit("DINOv3モデルの読み込みが完了しました")
            self.finished.emit(True, model, processor)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.progress.emit(f"エラー: {e}")
            self.progress.emit(f"詳細: {error_details[:500]}")
            print(f"Model loading error details:\n{error_details}")
            self.finished.emit(False, None, None)

class DirectoryProcessorThread(QThread):
    """ディレクトリ処理用のワーカースレッド"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    count_updated = pyqtSignal()

    def __init__(self, directory_path, db_path, feature_extractor, model_name=None):
        super().__init__()
        self.directory_path = directory_path
        self.db_path = db_path
        self.feature_extractor = feature_extractor
        self.model_name = model_name

    def run(self):
        try:
            self.progress.emit(f"フォルダを処理中: {self.directory_path}")

            # 画像ファイルを検索
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

                    # 既に存在するかチェック
                    cursor.execute("SELECT id FROM images WHERE file_hash = ?", (file_hash,))
                    if cursor.fetchone():
                        skipped_files += 1
                        processed_files += 1
                        continue

                    # 新規画像の場合のみ追加
                    if self.add_image_to_database(image_path, cursor):
                        added_files += 1

                    processed_files += 1
                    if processed_files % 5 == 0 or processed_files == total_files:
                        self.progress.emit(
                            f"処理中: {processed_files}/{total_files} "
                            f"(追加: {added_files}, スキップ: {skipped_files})"
                        )

                except Exception as e:
                    self.progress.emit(f"エラー ({os.path.basename(image_path)}): {e}")

            # モデル情報を設定
            if self.model_name and added_files > 0:
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_info (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                cursor.execute('SELECT model_name FROM model_info WHERE id = 1')
                if not cursor.fetchone():
                    cursor.execute('INSERT INTO model_info (id, model_name) VALUES (1, ?)', (self.model_name,))

            conn.commit()
            conn.close()

            self.finished.emit(
                f"フォルダの処理が完了しました。{processed_files}ファイル処理、"
                f"{added_files}ファイル追加、{skipped_files}ファイルスキップ。"
            )
            self.count_updated.emit()

        except Exception as e:
            self.finished.emit(f"ディレクトリ処理エラー: {e}")

    def calculate_file_hash(self, file_path):
        """ファイルハッシュを計算"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def add_image_to_database(self, file_path, cursor):
        """画像をデータベースに追加"""
        try:
            image = Image.open(file_path)
            file_hash = self.calculate_file_hash(file_path)

            # サムネイル作成
            thumbnail = image.copy()
            thumbnail.thumbnail((150, 150))
            buffer = io.BytesIO()
            thumbnail.save(buffer, format="JPEG")
            thumbnail_bytes = buffer.getvalue()

            # 画像情報を挿入
            cursor.execute(
                "INSERT INTO images (file_path, file_hash, thumbnail) VALUES (?, ?, ?)",
                (file_path, file_hash, thumbnail_bytes)
            )
            image_id = cursor.lastrowid

            # 特徴量を抽出
            feature_vector = self.feature_extractor.extract_features(image)
            if feature_vector is None:
                return False

            # 特徴量を挿入
            cursor.execute(
                "INSERT INTO features (image_id, feature_vector) VALUES (?, ?)",
                (image_id, feature_vector.tobytes())
            )

            return True
        except Exception as e:
            print(f"Error ({os.path.basename(file_path)}): {e}")
            return False

class ZipProcessorThread(QThread):
    """ZIP処理用のワーカースレッド"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    count_updated = pyqtSignal()

    def __init__(self, zip_path, db_path, feature_extractor, model_name=None):
        super().__init__()
        self.zip_path = zip_path
        self.db_path = db_path
        self.feature_extractor = feature_extractor
        self.model_name = model_name

    def run(self):
        temp_dir = None
        try:
            self.progress.emit(f"ZIPファイルを処理中: {os.path.basename(self.zip_path)}")

            # 一時ディレクトリを作成
            temp_dir = tempfile.mkdtemp(prefix="dinov3_zip_")

            # ZIPファイルを展開
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # ZIP内のファイル一覧を取得
                file_list = zip_ref.namelist()

                # 画像ファイルのみをフィルタリング
                image_files = [f for f in file_list
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))
                              and not f.startswith('__MACOSX/')  # macOSの隠しファイルを除外
                              and not os.path.basename(f).startswith('.')]  # 隠しファイルを除外

                if not image_files:
                    self.finished.emit("ZIPファイル内に有効な画像ファイルが見つかりませんでした。")
                    return

                total_files = len(image_files)
                processed_files = 0
                added_files = 0
                skipped_files = 0

                self.progress.emit(f"ZIPファイル内の画像ファイル数: {total_files}")

                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                for file_path in image_files:
                    try:
                        # ファイルを一時ディレクトリに展開
                        extracted_path = zip_ref.extract(file_path, temp_dir)

                        # ファイルハッシュを計算
                        file_hash = self.calculate_file_hash(extracted_path)

                        # 既に存在するかチェック
                        cursor.execute("SELECT id FROM images WHERE file_hash = ?", (file_hash,))
                        if cursor.fetchone():
                            skipped_files += 1
                            processed_files += 1
                            continue

                        # ZIPファイル内のパスを記録用のファイルパスとして使用
                        zip_file_path = f"{os.path.basename(self.zip_path)}:{file_path}"

                        # 新規画像の場合のみ追加
                        if self.add_image_to_database(extracted_path, zip_file_path, file_hash, cursor):
                            added_files += 1

                        processed_files += 1
                        if processed_files % 5 == 0 or processed_files == total_files:
                            self.progress.emit(
                                f"処理中: {processed_files}/{total_files} "
                                f"(追加: {added_files}, スキップ: {skipped_files})"
                            )

                    except Exception as e:
                        self.progress.emit(f"エラー ({os.path.basename(file_path)}): {e}")
                        processed_files += 1

                # モデル情報を設定
                if self.model_name and added_files > 0:
                    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_info (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        model_name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    ''')
                    cursor.execute('SELECT model_name FROM model_info WHERE id = 1')
                    if not cursor.fetchone():
                        cursor.execute('INSERT INTO model_info (id, model_name) VALUES (1, ?)', (self.model_name,))

                conn.commit()
                conn.close()

                self.finished.emit(
                    f"ZIPファイルの処理が完了しました。{processed_files}ファイル処理、"
                    f"{added_files}ファイル追加、{skipped_files}ファイルスキップ。"
                )
                self.count_updated.emit()

        except Exception as e:
            self.finished.emit(f"ZIPファイル処理エラー: {e}")
        finally:
            # 一時ディレクトリをクリーンアップ
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Failed to delete temporary directory: {e}")

    def calculate_file_hash(self, file_path):
        """ファイルハッシュを計算"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def add_image_to_database(self, file_path, zip_file_path, file_hash, cursor):
        """画像をデータベースに追加"""
        try:
            image = Image.open(file_path)

            # サムネイル作成
            thumbnail = image.copy()
            thumbnail.thumbnail((150, 150))
            buffer = io.BytesIO()
            thumbnail.save(buffer, format="JPEG")
            thumbnail_bytes = buffer.getvalue()

            # 画像情報を挿入（ZIPファイル内のパスを記録）
            cursor.execute(
                "INSERT INTO images (file_path, file_hash, thumbnail) VALUES (?, ?, ?)",
                (zip_file_path, file_hash, thumbnail_bytes)
            )
            image_id = cursor.lastrowid

            # 特徴量を抽出
            feature_vector = self.feature_extractor.extract_features(image)
            if feature_vector is None:
                return False

            # 特徴量を挿入
            cursor.execute(
                "INSERT INTO features (image_id, feature_vector) VALUES (?, ?)",
                (image_id, feature_vector.tobytes())
            )

            return True
        except Exception as e:
            print(f"Error ({os.path.basename(file_path)}): {e}")
            return False

class FeatureExtractor:
    """DINOv3特徴量抽出クラス"""
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DINOv3用の画像変換（ImageNetの標準値）
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image):
        """画像から特徴量を抽出"""
        try:
            # デバッグ: 入力画像の情報を出力
            print(f"\n=== Input Image Debug ===")
            print(f"Image mode: {image.mode}")
            print(f"Image size: {image.size}")
            # 画像の一部をハッシュ化して表示
            import hashlib
            img_bytes = image.tobytes()
            img_hash = hashlib.md5(img_bytes[:1000]).hexdigest()[:16]
            print(f"Image hash (first 1000 bytes): {img_hash}")

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 特徴量を抽出
            with torch.inference_mode():
                # プロセッサーで画像を前処理
                inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)

                # デバッグ: 変換後のテンソル情報
                pixel_values = inputs['pixel_values']
                print(f"Converted tensor shape: {pixel_values.shape}")
                print(f"Converted tensor mean: {pixel_values.mean().item():.6f}")
                print(f"Converted tensor std: {pixel_values.std().item():.6f}")

                # モデルで特徴量を抽出
                outputs = self.model(**inputs)

                # pooler_outputを使用（DINOv3の推奨方法）
                features = outputs.pooler_output.cpu().numpy()

            # 1次元配列に変換
            features = features.flatten()

            # デバッグ出力
            print(f"\n=== Feature Extraction Debug ===")
            print(f"Feature shape before normalization: {features.shape}")
            print(f"Norm before normalization: {np.linalg.norm(features):.6f}")

            # 正規化
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            print(f"Norm after normalization: {np.linalg.norm(features):.6f}")
            print(f"First 10 elements: {features[:10]}")

            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None

class SimilarImageWidget(QWidget):
    """類似画像表示ウィジェット"""
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 画像表示ラベル
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

        # 情報表示ラベル
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
        """画像データを設定"""
        if not image_data:
            self.clear()
            return

        # 画像を表示（アスペクト比を保持）
        pixmap = QPixmap()
        pixmap.loadFromData(image_data["thumbnail"])

        # アスペクト比を保持してスケール
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

        # 情報を表示
        similarity_percentage = image_data["similarity"] * 100
        file_path = image_data["file_path"]

        # ZIPファイル内の画像かどうかを判定してファイル名を表示
        if ':' in file_path:
            # ZIPファイル内の画像の場合
            zip_name, internal_path = file_path.split(':', 1)
            file_name = os.path.basename(internal_path)
            if len(file_name) > 12:
                file_name = file_name[:10] + "..."
            display_name = f"📦{file_name}"
        else:
            # 通常のファイルの場合
            file_name = os.path.basename(file_path)
            if len(file_name) > 12:
                file_name = file_name[:10] + "..."
            display_name = file_name

        info_text = f"{display_name}\nSimilarity: {similarity_percentage:.1f}%"

        # 類似度に応じて色分け
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
        """表示をクリア"""
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
    """ドラッグ＆ドロップエリア"""
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.current_pixmap = None
        self.is_image_displayed = False
        self.setup_ui()

    def setup_ui(self):
        """初期UIセットアップ"""
        if not self.is_image_displayed:
            self.setMinimumHeight(200)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setText("Drag and drop images, folders, or ZIP files here")
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
        """画像表示用のスタイルを復元"""
        self.setStyleSheet(f"""
            QLabel {{
                border: 2px solid {COLORS['primary']};
                border-radius: 12px;
                background-color: white;
                padding: 10px;
            }}
        """)

    def set_pixmap_from_pil(self, pil_image):
        """PIL画像から表示"""
        try:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # PIL画像をバイト配列に変換してQPixmapで読み込み
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            pixmap = QPixmap()
            if pixmap.loadFromData(img_buffer.getvalue()):
                self.display_pixmap(pixmap)
            else:
                self.show_error_message("画像の変換に失敗しました")

        except Exception as e:
            print(f"PIL image display error: {e}")
            import traceback
            traceback.print_exc()
            self.show_error_message("画像の表示に失敗しました")

    def display_pixmap(self, pixmap):
        """QPixmapを表示"""
        try:
            # 固定サイズでアスペクト比を保持してリサイズ
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
            self.show_error_message("画像の表示に失敗しました")

    def show_error_message(self, message):
        """エラーメッセージを表示"""
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
        """前の状態に復元"""
        if self.is_image_displayed and self.current_pixmap:
            self.setPixmap(self.current_pixmap)
            self.setText("")
            self.restore_image_display_style()
        else:
            self.setup_ui()

    def clear(self):
        """表示を完全にクリア"""
        self.current_pixmap = None
        self.is_image_displayed = False
        QLabel.clear(self)
        self.setText("Drag and drop images, folders, or ZIP files here")
        self.setup_ui()

class DINOv3ImageSearchApp(QMainWindow):
    """メインアプリケーションクラス"""

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
        """データベースの初期化"""
        self.db_files = []
        self.scan_db_files()

    def scan_db_files(self):
        """データベースファイルをスキャン"""
        self.db_files = []
        for db_file in glob.glob(os.path.join(mypath, "*.db")):
            self.db_files.append(os.path.basename(db_file))

        # デフォルトのデータベースファイルが存在しない場合は作成
        default_db_file = "image_features_dinov3.db"
        if default_db_file not in self.db_files:
            default_db_path = os.path.join(mypath, default_db_file)
            self.initialize_single_database(default_db_path)
            self.db_files.append(default_db_file)

    def initialize_single_database(self, db_path, model_name=None):
        """単一のデータベースを初期化"""
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

        # モデル情報を保存するテーブル
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_info (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            model_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # モデル情報が未設定の場合は設定
        if model_name:
            cursor.execute('SELECT model_name FROM model_info WHERE id = 1')
            existing = cursor.fetchone()
            if not existing:
                cursor.execute('INSERT INTO model_info (id, model_name) VALUES (1, ?)', (model_name,))

        conn.commit()
        conn.close()

    def get_database_model(self, db_path):
        """データベースに保存されているモデル名を取得"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # model_infoテーブルが存在するか確認
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_info'")
            if not cursor.fetchone():
                conn.close()
                return None

            cursor.execute('SELECT model_name FROM model_info WHERE id = 1')
            result = cursor.fetchone()
            conn.close()

            return result[0] if result else None
        except Exception as e:
            print(f"Model info retrieval error: {e}")
            return None

    def set_database_model(self, db_path, model_name):
        """データベースにモデル名を設定"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_info (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                model_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            cursor.execute('SELECT model_name FROM model_info WHERE id = 1')
            existing = cursor.fetchone()

            if existing:
                cursor.execute('UPDATE model_info SET model_name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1', (model_name,))
            else:
                cursor.execute('INSERT INTO model_info (id, model_name) VALUES (1, ?)', (model_name,))

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Model info setting error: {e}")

    def setup_ui(self):
        """UIのセットアップ"""
        model_type = "DINOv3" if USE_DINOV3 else "DINOv2"
        self.setWindowTitle(f"{model_type}Image Search Tool")
        self.setGeometry(100, 100, 1200, 800)

        # メインウィジェット
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # メインレイアウト
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ヘッダー部分
        self.setup_header(main_layout)

        # コントロール部分
        self.setup_controls(main_layout)

        # コンテンツ部分（スプリッター）
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        # 左側：Image Display Area
        self.setup_image_area(splitter)

        # 右側：Search Resultsエリア
        self.setup_results_area(splitter)

        # ステータスバー
        self.setup_status_bar()

    def setup_header(self, layout):
        """ヘッダー部分のセットアップ"""
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

        # ロゴ
        model_type = "DINOv3" if USE_DINOV3 else "DINOv2"
        logo_label = QLabel(model_type)
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

        # タイトル
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
        """コントロール部分のセットアップ"""
        controls_frame = QFrame()
        controls_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['card']};
                border-radius: 8px;
                padding: 10px;
            }}
        """)

        controls_layout = QVBoxLayout(controls_frame)

        # モデル選択行
        model_layout = QHBoxLayout()

        model_layout.addWidget(QLabel("Model:"))

        self.model_combo = QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS.keys())
        self.model_combo.setCurrentText(DEFAULT_MODEL)
        self.model_combo.currentTextChanged.connect(self.on_model_change)
        model_layout.addWidget(self.model_combo)

        model_layout.addStretch()

        controls_layout.addLayout(model_layout)

        # データベース管理行
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

        self.db_model_label = QLabel("")
        self.db_model_label.setStyleSheet(f"color: {COLORS['secondary']}; font-style: italic;")
        db_layout.addWidget(self.db_model_label)

        db_layout.addStretch()

        # データベース操作ボタン
        self.new_db_entry = QLineEdit()
        self.new_db_entry.setPlaceholderText("New DB Name")
        self.new_db_entry.setMaximumWidth(150)
        db_layout.addWidget(self.new_db_entry)

        create_db_btn = QPushButton("Create New")
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

        # ファイル操作行
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

        # プログレスバー
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
        """Image Display Areaのセットアップ"""
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

        # タイトル
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

        # ドロップエリア
        self.drop_area = DropAreaWidget()
        self.drop_area.files_dropped.connect(self.handle_dropped_files)
        image_layout.addWidget(self.drop_area, 1)

        splitter.addWidget(image_frame)

    def setup_results_area(self, splitter):
        """Search Resultsエリアのセットアップ"""
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

        # タイトル
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

        # 類似Image Display Area（2段×5列のグリッド）
        similar_frame = QFrame()
        similar_layout = QGridLayout(similar_frame)
        similar_layout.setContentsMargins(5, 5, 5, 5)
        similar_layout.setSpacing(10)

        # 10個の類似画像ウィジェットを作成
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
        """ステータスバーのセットアップ"""
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
        """全体的なスタイルのセットアップ"""
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
                color: {COLORS['text']};
                min-width: 150px;
            }}

            QComboBox:focus {{
                border-color: {COLORS['primary']};
            }}

            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}

            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {COLORS['text']};
                margin-right: 5px;
            }}

            QComboBox QAbstractItemView {{
                background-color: white;
                color: {COLORS['text']};
                selection-background-color: {COLORS['primary']};
                selection-color: white;
                border: 1px solid {COLORS['border']};
                outline: none;
            }}

            QComboBox QAbstractItemView::item {{
                padding: 5px;
                min-height: 25px;
            }}

            QComboBox QAbstractItemView::item:hover {{
                background-color: {COLORS['secondary']};
                color: white;
            }}

            QComboBox QAbstractItemView::item:selected {{
                background-color: {COLORS['primary']};
                color: white;
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

    def load_models(self, model_name=None):
        """モデルの読み込み"""
        self.model_thread = ModelLoaderThread(model_name)
        self.model_thread.progress.connect(self.update_status)
        self.model_thread.finished.connect(self.on_model_loaded)
        self.model_thread.start()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

    def on_model_change(self, model_display_name):
        """モデル変更時の処理"""
        model_name = AVAILABLE_MODELS.get(model_display_name)
        if not model_name:
            return

        reply = QMessageBox.question(
            self,
            "Model Change",
            f"Load {model_display_name}?\n\nNote: Larger models take longer to load and use more memory.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.model_loaded = False
            self.feature_extractor = None
            self.load_models(model_name)

    def on_model_loaded(self, success, model, processor):
        """モデル読み込み完了時の処理"""
        self.progress_bar.setVisible(False)
        model_name = "DINOv3" if USE_DINOV3 else "DINOv2"

        if success:
            self.feature_extractor = FeatureExtractor(model, processor)
            self.model_loaded = True
            self.update_status(f"{model_name}モデルの読み込みが完了しました。")
            self.update_image_count()
        else:
            self.update_status(f"{model_name}モデルの読み込みに失敗しました。")
            QMessageBox.critical(self, "Error", f"Failed to load {model_name} model.")

    def update_status(self, message):
        """ステータスメッセージをRefresh"""
        self.status_bar.showMessage(message)

    def update_image_count(self):
        """画像枚数をRefresh"""
        if not self.db_combo.currentText():
            self.image_count_label.setText("(0 images)")
            self.db_model_label.setText("")
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

            # データベースのモデル情報を表示
            db_model = self.get_database_model(db_path)
            if db_model:
                # モデル名からわかりやすい表示名を取得
                display_name = None
                for name, model_id in AVAILABLE_MODELS.items():
                    if model_id == db_model:
                        display_name = name
                        break

                if display_name:
                    self.db_model_label.setText(f"[{display_name}]")
                else:
                    self.db_model_label.setText(f"[{db_model}]")
            else:
                self.db_model_label.setText("[Model not set]")

        except Exception as e:
            self.image_count_label.setText("(Error)")
            self.db_model_label.setText("")
            print(f"Image count retrieval error: {e}")

    def on_database_change(self, db_name):
        """データベース変更時の処理"""
        self.update_image_count()
        for widget in self.similar_widgets:
            widget.clear()

    def handle_dropped_files(self, files):
        """ドロップされたファイルを処理"""
        if not self.model_loaded:
            QMessageBox.information(self, "Information", "DINOv3 model is still loading. Please wait.")
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
        """画像ファイルを選択"""
        if not self.model_loaded:
            QMessageBox.information(self, "Information", "DINOv3 model is still loading.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "画像ファイル (*.jpg *.jpeg *.png *.bmp *.gif *.webp)"
        )

        if file_path:
            self.process_image_file(file_path)

    def select_folder(self):
        """Select Folder"""
        if not self.model_loaded:
            QMessageBox.information(self, "Information", "DINOv3 model is still loading.")
            return

        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")

        if folder_path:
            self.process_directory(folder_path)

    def select_zip(self):
        """ZIPファイルを選択"""
        if not self.model_loaded:
            QMessageBox.information(self, "Information", "DINOv3 model is still loading.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "ZIPファイルを選択", "",
            "ZIPファイル (*.zip)"
        )

        if file_path:
            self.process_zip_file(file_path)

    def process_image_file(self, file_path):
        """単一画像ファイルを処理"""
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
            self.update_status(f"画像処理エラー: {e}")
            QMessageBox.warning(self, "Warning", f"Error occurred during image processing:\n{str(e)[:100]}...")

    def process_zip_file(self, zip_path):
        """ZIPファイルを処理"""
        if not self.feature_extractor:
            QMessageBox.information(self, "Information", "Model is still loading.")
            return

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                pass
        except zipfile.BadZipFile:
            QMessageBox.warning(self, "エラー", "無効なZIPファイルです。")
            return
        except Exception as e:
            QMessageBox.warning(self, "エラー", f"ZIPファイルの読み込みに失敗しました: {e}")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        db_path = os.path.join(mypath, self.db_combo.currentText())
        current_model_id = AVAILABLE_MODELS.get(self.model_combo.currentText())

        self.zip_thread = ZipProcessorThread(
            zip_path, db_path, self.feature_extractor, current_model_id
        )
        self.zip_thread.progress.connect(self.update_status)
        self.zip_thread.finished.connect(self.on_zip_processed)
        self.zip_thread.count_updated.connect(self.update_image_count)
        self.zip_thread.start()

    def on_zip_processed(self, message):
        """ZIP処理完了時の処理"""
        self.progress_bar.setVisible(False)
        self.update_status(message)

    def extract_and_search_features(self, image):
        """特徴量抽出と検索を実行"""
        try:
            self.update_status("特徴量を抽出中...")

            feature_vector = self.feature_extractor.extract_features(image)

            if feature_vector is None:
                self.update_status("特徴量の抽出に失敗しました。")
                QMessageBox.warning(self, "Warning", "Failed to extract features.")
                return

            self.search_similar_images(feature_vector)

        except Exception as e:
            print(f"Feature extraction error: {e}")
            self.update_status(f"Feature extraction error: {e}")
            QMessageBox.warning(self, "Warning", f"Error occurred during feature extraction:\n{str(e)[:100]}...")

    def search_similar_images(self, query_features):
        """類似画像を検索"""
        try:
            db_path = os.path.join(mypath, self.db_combo.currentText())
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT image_id, feature_vector FROM features")
            results = cursor.fetchall()

            if not results:
                self.update_status("データベースに画像が登録されていません。")
                for widget in self.similar_widgets:
                    widget.clear()
                conn.close()
                return

            # デバッグ: クエリ特徴量の情報を出力
            print(f"\n=== Similarity Calculation Debug ===")
            print(f"Query feature shape: {query_features.shape}")
            print(f"Query feature norm: {np.linalg.norm(query_features):.6f}")
            print(f"Query feature first 10 elements: {query_features[:10]}")

            similarities = []
            for result_id, result_bytes in results:
                result_features = np.frombuffer(result_bytes, dtype=np.float32)
                # 特徴量は既に正規化済みなので、内積のみで類似度を計算
                similarity = np.dot(query_features, result_features)
                similarities.append((result_id, similarity))

                # デバッグ: 最初の3件の類似度計算を詳しく出力
                if len(similarities) <= 3:
                    print(f"\nDB Image ID {result_id}:")
                    print(f"  DB feature shape: {result_features.shape}")
                    print(f"  DB feature norm: {np.linalg.norm(result_features):.6f}")
                    print(f"  Similarity: {similarity:.6f}")

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
            self.update_status(f"類似画像検索が完了しました。トップ{len(similar_images)}を表示中。")

        except Exception as e:
            self.update_status(f"類似画像検索エラー: {e}")
            QMessageBox.warning(self, "Warning", f"Error occurred during similar image search:\n{str(e)[:100]}...")

    def display_similar_images(self, similar_images):
        """類似画像を表示"""
        for i, widget in enumerate(self.similar_widgets):
            if i < len(similar_images):
                widget.set_image_data(similar_images[i])
            else:
                widget.clear()

    def process_directory(self, directory_path):
        """ディレクトリを処理"""
        if not self.feature_extractor:
            QMessageBox.information(self, "情報", "モデルがまだ読み込まれていません。")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        db_path = os.path.join(mypath, self.db_combo.currentText())
        current_model_id = AVAILABLE_MODELS.get(self.model_combo.currentText())

        self.dir_thread = DirectoryProcessorThread(
            directory_path, db_path, self.feature_extractor, current_model_id
        )
        self.dir_thread.progress.connect(self.update_status)
        self.dir_thread.finished.connect(self.on_directory_processed)
        self.dir_thread.count_updated.connect(self.update_image_count)
        self.dir_thread.start()

    def on_directory_processed(self, message):
        """ディレクトリ処理完了時の処理"""
        self.progress_bar.setVisible(False)
        self.update_status(message)

    def create_new_database(self):
        """新規データベースを作成"""
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

        QMessageBox.information(self, "Information", f"Database '{db_name}' created.")

    def clear_database(self):
        """データベースを消去"""
        if not self.db_combo.currentText():
            QMessageBox.warning(self, "Warning", "No database selected.")
            return

        reply = QMessageBox.question(
            self, "Confirmation",
            f"Clear contents of database '{self.db_combo.currentText()}'.\n"
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

                QMessageBox.information(self, "Information", f"Database '{self.db_combo.currentText()}' cleared.")
                self.update_image_count()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error occurred while clearing database: {e}")

    def delete_database(self):
        """データベースファイルを削除"""
        if not self.db_combo.currentText():
            QMessageBox.warning(self, "Warning", "No database selected.")
            return

        current_db = self.db_combo.currentText()

        reply = QMessageBox.question(
            self, "Confirmation",
            f"Completely delete database file '{current_db}'.\n"
            "This operation cannot be undone. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                db_path = os.path.join(mypath, current_db)
                os.remove(db_path)
                self.refresh_db_list()

                QMessageBox.information(self, "Information", f"Database file '{current_db}' deleted.")

                if self.db_files:
                    self.db_combo.setCurrentText(self.db_files[0])
                    self.update_image_count()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error occurred while deleting database: {e}")

    def clear_image_display(self):
        """Image Display Areaをクリア"""
        self.drop_area.clear()
        self.current_image = None
        self.current_image_path = None
        self.add_to_db_btn.setEnabled(False)

        for widget in self.similar_widgets:
            widget.clear()
        self.update_status("画像表示をクリアしました。")

    def add_current_image_to_db(self):
        """現在表示中の画像をデータベースに追加"""
        if not self.current_image or not self.model_loaded:
            QMessageBox.warning(self, "Warning", "No image to add or model not loaded.")
            return

        if not self.db_combo.currentText():
            QMessageBox.warning(self, "Warning", "No database selected.")
            return

        # モデル不一致チェック
        db_path = os.path.join(mypath, self.db_combo.currentText())
        db_model = self.get_database_model(db_path)
        current_model = self.model_combo.currentText()
        current_model_id = AVAILABLE_MODELS.get(current_model)

        if db_model and db_model != current_model_id:
            # データベースのモデル名を表示名に変換
            db_model_display = None
            for name, model_id in AVAILABLE_MODELS.items():
                if model_id == db_model:
                    db_model_display = name
                    break

            reply = QMessageBox.warning(
                self,
                "Model Mismatch Warning",
                f"Database was created with {db_model_display or db_model},\n"
                f"but currently using {current_model}.\n\n"
                f"Features extracted with different models are not compatible,\n"
                f"and search results will not be accurate.\n\n"
                f"Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
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

            self.update_status("特徴量を抽出中...")
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

            # モデル情報を設定（未設定の場合のみ）
            # model_infoテーブルが存在しない場合は作成
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_info (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                model_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            cursor.execute('SELECT model_name FROM model_info WHERE id = 1')
            existing = cursor.fetchone()
            if not existing and current_model_id:
                print(f"[DEBUG] Inserting model info: {current_model_id}")
                cursor.execute('INSERT INTO model_info (id, model_name) VALUES (1, ?)', (current_model_id,))

            conn.commit()
            conn.close()

            file_name = os.path.basename(file_path) if file_path else "dropped_image"
            QMessageBox.information(self, "Success", f"Image \"{file_name}\" added to database.")

            self.update_image_count()
            self.update_status(f"Image \"{file_name}\" added to database.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error occurred while adding to database:\n{str(e)}")
            print(f"DB add error: {e}")
            import traceback
            traceback.print_exc()

    def calculate_file_hash(self, file_path):
        """ファイルハッシュを計算"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def refresh_db_list(self):
        """データベースリストをRefresh"""
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
