import sys
import os
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog, QPushButton

OUTPUT_DIR = "maps_cropped_100px"
IMG_A = "a"
IMG_B = "b"
SIZE = 100

class ImageViewer(QGraphicsView):
    def __init__(self, image_path, name_label, parent_callback):
        super().__init__()
        self.name_label = name_label
        self.parent_callback = parent_callback
        self.scene = QGraphicsScene(self)
        self.pixmap = QPixmap(image_path)
        self.image_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.image_item)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0

        self._mouse_pressed = False
        self._mouse_moved = False
        self._last_mouse_pos = None
        self._move_threshold = 5

        self.selected_point = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom_step = 0.01
        old_scale = self.scale_factor

        if delta > 0 and self.scale_factor < self.max_scale:
            self.scale_factor += zoom_step
        elif delta < 0 and self.scale_factor > self.min_scale:
            self.scale_factor -= zoom_step
        else:
            return

        scale_change = self.scale_factor / old_scale
        self.scale(scale_change, scale_change)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._mouse_pressed = True
            self._mouse_moved = False
            self._last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self._mouse_pressed:
            if (event.pos() - self._last_mouse_pos).manhattanLength() > self._move_threshold:
                self._mouse_moved = True
                delta = event.pos() - self._last_mouse_pos
                self._last_mouse_pos = event.pos()
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if not self._mouse_moved:
                scene_pos = self.mapToScene(event.pos())
                self.selected_point = (scene_pos.x(), scene_pos.y())
                print(f"[{self.name_label}] Selected point: {self.selected_point}")
                self.parent_callback(self.name_label, self.selected_point)
            self._mouse_pressed = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual Image Viewer with Zoom, Pan, Cropping, Save on Pair")
        self.setGeometry(100, 100, 1300, 700)

        self.point_a = None
        self.point_b = None
        self.save_counter = 1

        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            if len(files) == 2:
                self.init_ui(files[0], files[1])
            else:
                print("Please select exactly 2 images.")
                sys.exit(1)
        else:
            sys.exit(1)

    def init_ui(self, img1, img2):
        main_layout = QVBoxLayout()
        viewer_layout = QHBoxLayout()

        self.viewer_a = ImageViewer(img1, 'A', self.point_selected)
        self.viewer_b = ImageViewer(img2, 'B', self.point_selected)

        viewer_layout.addWidget(self.viewer_a)
        viewer_layout.addWidget(self.viewer_b)

        reset_btn = QPushButton("Reset Points")
        reset_btn.clicked.connect(self.reset_points)

        main_layout.addLayout(viewer_layout)
        main_layout.addWidget(reset_btn)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def point_selected(self, label, point):
        if label == 'A':
            self.point_a = point
        elif label == 'B':
            self.point_b = point

        if self.point_a and self.point_b:
            self.save_crops()
            self.point_a = None
            self.point_b = None
            self.viewer_a.selected_point = None
            self.viewer_b.selected_point = None

    def save_crops(self):
        crop_size = SIZE
        half = crop_size / 2

        # Create folders
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Crop Image A
        xA, yA = self.point_a
        rectA = QRectF(xA - half, yA - half, crop_size, crop_size).intersected(QRectF(0, 0, self.viewer_a.pixmap.width(), self.viewer_a.pixmap.height()))
        if not rectA.isEmpty():
            croppedA = self.viewer_a.pixmap.copy(rectA.toRect())
            pathA = os.path.join(OUTPUT_DIR, f"{IMG_A}_{self.save_counter:03d}.png")
            croppedA.save(pathA)
            print(f"Saved: {pathA}")
        else:
            print("Point A out of bounds")

        # Crop Image B
        xB, yB = self.point_b
        rectB = QRectF(xB - half, yB - half, crop_size, crop_size).intersected(QRectF(0, 0, self.viewer_b.pixmap.width(), self.viewer_b.pixmap.height()))
        if not rectB.isEmpty():
            croppedB = self.viewer_b.pixmap.copy(rectB.toRect())
            pathB = os.path.join(OUTPUT_DIR, f"{IMG_B}_{self.save_counter:03d}.png")
            croppedB.save(pathB)
            print(f"Saved: {pathB}")
        else:
            print("Point B out of bounds")

        self.save_counter += 1

    def reset_points(self):
        print("Reset points")
        self.point_a = None
        self.point_b = None
        self.viewer_a.selected_point = None
        self.viewer_b.selected_point = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
