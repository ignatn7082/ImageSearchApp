import sys

from os import environ, path, walk, remove, name
from subprocess import run
from pickle import load, dump, UnpicklingError
import numpy as np
import torch
from PIL import Image
import clip
from sklearn.metrics.pairwise import cosine_similarity

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QGridLayout, QMessageBox, QSpinBox, QTabWidget, QLineEdit, QScrollArea,
    QFrame, QProgressDialog
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt


environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def extract_clip_vector(model, preprocess, input_data, is_text=False, device="cpu"):
    if is_text:
        text_input = clip.tokenize([input_data]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
        vector = text_features / text_features.norm(dim=-1, keepdim=True)
    else:
        image = preprocess(Image.open(input_data).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        vector = image_features / image_features.norm(dim=-1, keepdim=True)
    return vector.cpu().numpy().flatten()

def calculate_distance(vectors, query_vector):
    query_vector = query_vector.reshape(1, -1)
    distance = -cosine_similarity(vectors, query_vector)
    return distance

def open_folder(folder_path):
    if name == 'nt':
        run(['explorer', folder_path])

def on_click(event, image_path):
    if image_path:
        folder_path = path.dirname(image_path)
        print("Clicked image:", image_path)
        print("Opening folder:", folder_path)
        
        open_folder(folder_path)

def get_all_image_paths(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    paths = []
    for root, dirs, files in walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                paths.append(path.join(root, file))
    return paths

device = "cuda" if torch.cuda.is_available() else "cpu"

if sys.stdout is None:
    sys.stdout = sys.__stdout__
if sys.stderr is None:
    sys.stderr = sys.__stderr__
model, preprocess = clip.load("ViT-B/32", download_root="clip_model", jit=False)


class ImageSearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Search App")
        self.setWindowIcon(QIcon("res/search.png"))  
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("""
    QMainWindow {
        background-color: #f5f5f5;
    }

    QLabel {
        font-size: 18px;
        font-family: 'Roboto', 'Segoe UI';
        color: #333333;
    }

    QLineEdit, QSpinBox {
        font-size: 16px;
        font-family: 'Roboto', 'Segoe UI';
        padding: 6px 10px;
        border: 1px solid #ccc;
        border-radius: 6px;
        background-color: white;
    }

    QPushButton {
        font-size: 16px;
        font-family: 'Roboto', 'Segoe UI';
        color: white;
        background-color: #2196F3; /* Material Blue */
        border-radius: 8px;
        border: 1px solid black ;
        padding: 8px 16px;

    }

    QPushButton:hover {
        background-color: #1976D2; /* Darker blue */
    }

    QTabWidget::pane {
        border: none;
        background: white;
        border-radius: 8px;
    }

    QTabBar::tab {
        font-size: 16px;
        background: #eeeeee;
        color: #333;
        padding: 8px 16px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        margin-right: 2px;
        font-family: 'Roboto', 'Segoe UI';
    }

    QTabBar::tab:selected {
        background: #2196F3;
        color: white;
    }
""")


        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.image_tab = QWidget()
        self.text_tab = QWidget()

        self.tabs.addTab(self.image_tab, QIcon("res/image_tab.png"), "Hình ảnh")
        self.tabs.addTab(self.text_tab, QIcon("res/text_tab.png"), "Văn bản")

        self.init_image_tab()
        self.init_text_tab()


    
    def init_image_tab(self):
        layout = QVBoxLayout()

       
        title = QLabel("🔍 TÌM KIẾM BẰNG HÌNH ẢNH")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        self.setStyleSheet(""" QWidget {
        background-color: aliceblue;

    };
            QLabel {
                border: 2px dashed #aaa;
                background-color: #fafafa;
                border-radius: 12px;
                font-size: 20px;
                color: #888;
            }
        """)
        # Ảnh xem trước
        self.image_frame = QLabel("Xem trước ảnh")
        self.image_frame.setFixedSize(300, 300)
        self.image_frame.setAlignment(Qt.AlignCenter)
        self.image_frame.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                background-color: #fafafa;
                border-radius: 12px;
                font-size: 20px;
                color: #888;
            }
        """)
        layout.addWidget(self.image_frame, alignment=Qt.AlignCenter)

        
        control_layout = QHBoxLayout()
        control_layout.setSpacing(12)

        self.num_results_input = QSpinBox()
        self.num_results_input.setStyleSheet("background-color: #fafafa;")
      
        self.num_results_input.setRange(1, 10)
        self.num_results_input.setValue(5)
        self.num_results_input.setFixedSize(50,30)  

        num_label = QLabel(" Số ảnh hiển thị:")
        num_label.setStyleSheet("font-size: 18px; font-weight: bold;")  
        control_layout.addWidget(num_label)
        control_layout.addWidget(self.num_results_input)

        button_style = "padding: 6px 12px; border-radius: 8px; font-weight: bold;"

        self.upload_button = QPushButton("Tải ảnh lên")
        self.upload_button.setStyleSheet(button_style + "background-color: #2196F3; color: white;")
        self.upload_button.clicked.connect(self.load_image)
        control_layout.addWidget(self.upload_button)

        self.search_button = QPushButton("Tìm kiếm")
        self.search_button.setStyleSheet(button_style + "background-color: #2196F3; color: white;")
        self.search_button.clicked.connect(self.search_image)
        control_layout.addWidget(self.search_button)

        self.update_db_button = QPushButton("Cập nhật CSDL ảnh")
        self.update_db_button.setStyleSheet(button_style + "background-color: #2196F3; color: white;")
        self.update_db_button.clicked.connect(self.update_database)
        control_layout.addWidget(self.update_db_button)

        self.delete_db_button = QPushButton("Xoá CSDL ảnh")
        self.delete_db_button.setStyleSheet(button_style + "background-color: #f44336; color: white;")
        self.delete_db_button.clicked.connect(self.delete_database)
        control_layout.addWidget(self.delete_db_button)

        layout.addLayout(control_layout)

        result_label = QLabel("Kết quả tìm kiếm:")
        result_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(result_label)
        self.result_area = QScrollArea()
        self.result_area.setWidgetResizable(True)
        self.result_widget = QWidget()
        self.results_layout = QGridLayout(self.result_widget)
        self.result_area.setWidget(self.result_widget)
        layout.addWidget(self.result_area)

        self.image_tab.setLayout(layout)
        self.image_path = None
        
    def init_text_tab(self):
        layout = QVBoxLayout()

        title = QLabel("🔎 TÌM KIẾM BẰNG VĂN BẢN")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Thanh tìm kiếm 
        self.text_input = QLineEdit()
        self.text_input.setFixedWidth(int(self.width() * 0.4))
        self.text_input.setPlaceholderText("Nhập từ khóa tìm kiếm...")
        self.text_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                font-size: 20px;
                border: 1px solid #ccc;
                border-radius: 10px;
                background-color: #fafafa;
            }
        """)

        input_layout = QHBoxLayout()
        input_layout.addStretch()
        input_layout.addWidget(self.text_input)
        input_layout.addStretch()
        layout.addLayout(input_layout)

        # Các nút điều khiển
        control_layout = QHBoxLayout()
        control_layout.setSpacing(12)

        self.num_results_text_input = QSpinBox()
        self.num_results_text_input.setStyleSheet("background-color: white;")  
        self.num_results_text_input.setRange(1, 10)
        self.num_results_text_input.setValue(5)
        self.num_results_text_input.setFixedSize(50,30)  
        num_label_text = QLabel(" Số ảnh hiển thị:")
        num_label_text.setStyleSheet("font-size: 18px; font-weight: bold;")  
        control_layout.addWidget(num_label_text)
        control_layout.addWidget(self.num_results_text_input)

        button_style = "padding: 6px 12px; border-radius: 8px; font-weight: bold;"

        self.search_text_button = QPushButton("Tìm kiếm")
        self.search_text_button.setStyleSheet(button_style + "background-color: #2196F3; color: white;")
        self.search_text_button.clicked.connect(self.search_text)
        control_layout.addWidget(self.search_text_button)

        self.update_db_text_button = QPushButton("Cập nhật CSDL ảnh")
        self.update_db_text_button.setStyleSheet(button_style + "background-color: #2196F3; color: white;")
        self.update_db_text_button.clicked.connect(self.update_database)
        control_layout.addWidget(self.update_db_text_button)

        self.delete_db_text_button = QPushButton("Xoá CSDL ảnh")
        self.delete_db_text_button.setStyleSheet(button_style + "background-color: #f44336; color: white;")
        self.delete_db_text_button.clicked.connect(self.delete_database)
        control_layout.addWidget(self.delete_db_text_button)

        layout.addLayout(control_layout)

        result_label_text = QLabel("Kết quả tìm kiếm:")
        result_label_text.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(result_label_text)
        self.text_result_area = QScrollArea()
        self.text_result_area.setWidgetResizable(True)
        self.text_result_widget = QWidget()
        self.text_results_layout = QGridLayout(self.text_result_widget)
        self.text_result_area.setWidget(self.text_result_widget)
        layout.addWidget(self.text_result_area)

        self.text_tab.setLayout(layout)


    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_frame.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def search_image(self):
        if self.image_path is None:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn ảnh trước khi tìm kiếm.")
            return

        self.clear_results(self.results_layout)

        try:
            with open("vectors_image_3.pkl", "rb") as f:
                vectors = load(f)
            with open("paths_image_3.pkl", "rb") as f:
                paths = load(f)

            # Kiểm tra nếu vectors hoặc paths rỗng
            if len(paths) == 0 or len(vectors) == 0:
    
                QMessageBox.warning(self, "Cảnh báo", "Dữ liệu ảnh hoặc vector đang trống. Vui lòng cập nhật cơ sở dữ liệu trước.")
                return

        except (FileNotFoundError, EOFError, UnpicklingError) as e:
            QMessageBox.warning(self, "Cảnh báo", "Dữ liệu ảnh hoặc vector đang trống. Vui lòng cập nhật cơ sở dữ liệu trước.")
            return

        image_vector = extract_clip_vector(model, preprocess, self.image_path, is_text=False, device=device)
        distance = calculate_distance(vectors, image_vector)
        ids = np.argsort(distance.flatten())[:self.num_results_input.value()]

        for i, image_id in enumerate(ids):
            self.display_result_image(paths[image_id], self.results_layout, i)


    def search_text(self):
        query = self.text_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập từ khóa tìm kiếm.")
            return
        self.clear_results(self.text_results_layout)

        try:
            with open("vectors_image_3.pkl", "rb") as f:
                vectors = load(f)
            with open("paths_image_3.pkl", "rb") as f:
                paths = load(f)

            # Kiểm tra nếu vectors hoặc paths rỗng
            if len(paths) == 0 or len(vectors) == 0:
    
                QMessageBox.warning(self, "Cảnh báo", "Dữ liệu ảnh hoặc vector đang trống. Vui lòng cập nhật cơ sở dữ liệu trước.")
                return
        except (FileNotFoundError, EOFError, UnpicklingError) as e:
            QMessageBox.warning(self, "Cảnh báo", "Dữ liệu ảnh hoặc vector đang trống. Vui lòng cập nhật cơ sở dữ liệu trước.")
            return

        text_vector = extract_clip_vector(model, preprocess, query, is_text=True, device=device)
        distance = calculate_distance(vectors, text_vector)
        ids = np.argsort(distance.flatten())[:self.num_results_text_input.value()]

        for i, image_id in enumerate(ids):
            self.display_result_image(paths[image_id], self.text_results_layout, i)

    def display_result_image(self, image_path, layout, index):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 10px;")
        vbox = QVBoxLayout(frame)

        label = ClickableLabel(image_path) 
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setAlignment(Qt.AlignCenter)

        text = QLabel(path.basename(image_path))
        text.setAlignment(Qt.AlignCenter)
        text.setWordWrap(True)
        text.setStyleSheet("color: #333; font-size: 11px;")

        vbox.addWidget(label)
        vbox.addWidget(text)
        layout.addWidget(frame, index // 5, index % 5)


    def clear_results(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def update_database(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa ảnh")
        if not folder:
            return

        progress_dialog = QProgressDialog("Đang cập nhật cơ sở dữ liệu...", "Hủy", 0, 100, self)
        progress_dialog.setWindowTitle("Cập nhật CSDL")
        progress_dialog.setFixedSize(300, 100)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setValue(0)
        progress_dialog.setLabelText("Đang xử lý ảnh: 0%")
        progress_dialog.show()

        image_paths = get_all_image_paths(folder)

        # Tải dữ liệu cũ nếu có
        try:
            with open("vectors_image_3.pkl", "rb") as f:
                old_vectors = load(f)
            with open("paths_image_3.pkl", "rb") as f:
                old_paths = load(f)
        except:
            old_vectors = np.empty((0, 512))  # 512 là chiều vector của CLIP
            old_paths = []

        vectors = []
        valid_paths = []

        total_images = len(image_paths)
        processed = 0

        for i, path in enumerate(image_paths):
            if progress_dialog.wasCanceled():
                break

            # Bỏ qua ảnh đã xử lý
            if path in old_paths:
                continue

            try:
                vector = extract_clip_vector(model, preprocess, path, is_text=False, device=device)
                vectors.append(vector)
                valid_paths.append(path)
            except Exception as e:
                print(f"Không thể xử lý ảnh: {path}, lỗi: {e}")

            processed += 1
            progress_percentage = int((i + 1) / total_images * 100)
            progress_dialog.setValue(progress_percentage)
            progress_dialog.setLabelText(f"Đang xử lý ảnh: {progress_percentage}%")
            QApplication.processEvents()

        # Gộp dữ liệu cũ với mới
        if vectors:
            old_vectors = np.array(old_vectors)
            
            # Kiểm tra xem old_vectors có rỗng không
            if old_vectors.shape[0] == 0:  # Nếu old_vectors rỗng (0 dòng)
                all_vectors = np.array(vectors)  # Gán trực tiếp vectors vào all_vectors
            else:
                all_vectors = np.vstack([old_vectors, np.array(vectors)])  # Ghép old_vectors và vectors



            # all_vectors = np.vstack([old_vectors, np.array(vectors)])
            all_paths = old_paths + valid_paths

            with open("vectors_image_3.pkl", "wb") as f:
                dump(all_vectors, f)
            with open("paths_image_3.pkl", "wb") as f:
                dump(all_paths, f)

        progress_dialog.setValue(100)
        progress_dialog.setLabelText("Cập nhật hoàn tất!")
        QMessageBox.information(self, "Hoàn tất", f"Đã thêm {len(valid_paths)} ảnh mới vào CSDL.")


    def delete_database(self):
        reply = QMessageBox.question(
            self, "Xác nhận xoá", "Bạn có chắc chắn muốn xoá cơ sở dữ liệu ảnh?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                if path.exists("vectors_image_3.pkl"):
                    remove("vectors_image_3.pkl")
                if path.exists("paths_image_3.pkl"):
                    remove("paths_image_3.pkl")
                QMessageBox.information(self, "Thành công", "Đã xoá cơ sở dữ liệu ảnh.")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể xoá CSDL: {str(e)}")


class ClickableLabel(QLabel):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = path.normpath(image_path)

    def mousePressEvent(self, event):
        if self.image_path:
            folder_path = path.dirname(self.image_path)
            # print(" Clicked:", self.image_path)
            # print(" Opening folder:", folder_path)
            open_folder(folder_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSearchApp()
    window.show()
    sys.exit(app.exec_())
