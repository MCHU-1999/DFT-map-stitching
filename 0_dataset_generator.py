import sys
import os
import random
import math
import json
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QPainter, QTransform, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QGraphicsView, QGraphicsScene, 
                             QGraphicsPixmapItem, QFileDialog, QPushButton, 
                             QLabel, QSpinBox, QCheckBox, QGroupBox, QFormLayout,
                             QDialog, QRadioButton, QButtonGroup, QDialogButtonBox)

OUTPUT_DIR_TEST = "data_test/rename_me_after_create"
OUTPUT_DIR_EVAL = "data_eval/rename_me_after_create"
IMG_A = "a"
IMG_B = "b"
IMG_ORIGINAL = "original"
IMG_TRANSFORMED = "transformed"
CROP_SIZE = 100
MAX_TRANSLATION = 25
MAX_ROTATION = 180

class ModeSelectionDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Generator - Select Mode")
        self.setModal(True)
        self.resize(400, 250)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Choose Dataset Generation Mode")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Mode selection
        self.button_group = QButtonGroup()
        
        # Test mode
        test_mode = QRadioButton("Test Mode")
        test_mode.setChecked(True)  # Default selection
        test_desc = QLabel("• Select corresponding points on 2 images\n• Generate paired crops for testing registration algorithms\n• Output: a_XXX.png and b_XXX.png")
        test_desc.setStyleSheet("QLabel { margin-left: 20px; color: #666; }")
        
        # Evaluation mode  
        eval_mode = QRadioButton("Evaluation Mode")
        eval_desc = QLabel("• Click points on 1 image\n• Generate original + transformed pairs with known transformations\n• Output: a_XXX.png and b_XXX.png")
        eval_desc.setStyleSheet("QLabel { margin-left: 20px; color: #666; }")
        
        self.button_group.addButton(test_mode, 0)
        self.button_group.addButton(eval_mode, 1)
        
        layout.addWidget(QLabel(""))  # Spacer
        layout.addWidget(test_mode)
        layout.addWidget(test_desc)
        layout.addWidget(QLabel(""))  # Spacer
        layout.addWidget(eval_mode)
        layout.addWidget(eval_desc)
        layout.addWidget(QLabel(""))  # Spacer
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_selected_mode(self):
        return self.button_group.checkedId()  # 0 = test, 1 = evaluation

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
        zoom_step = 0.005  # Slower zoom (was 0.01)
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
        
        # Show mode selection dialog
        mode_dialog = ModeSelectionDialog()
        if mode_dialog.exec_() == QDialog.Accepted:
            self.mode = mode_dialog.get_selected_mode()  # 0 = test, 1 = evaluation
        else:
            sys.exit(0)
        
        self.save_counter = 1
        self.point_a = None
        self.point_b = None
        self.ground_truth_data = []  # Store ground truth for evaluation mode
        
        if self.mode == 0:  # Test mode
            self.init_test_mode()
        else:  # Evaluation mode
            self.init_evaluation_mode()

    def init_test_mode(self):
        """Initialize test mode (2 images input)"""
        self.setWindowTitle("Dataset Generator - Test Mode (2 Images)")
        self.setGeometry(100, 100, 1300, 700)
        
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.jpeg)")
        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            if len(files) == 2:
                self.init_test_ui(files[0], files[1])
            else:
                print("Please select exactly 2 images.")
                sys.exit(1)
        else:
            sys.exit(1)

    def init_test_ui(self, img1, img2):
        """Create UI for test mode"""
        main_layout = QVBoxLayout()
        viewer_layout = QHBoxLayout()

        self.viewer_a = ImageViewer(img1, 'A', self.point_selected_test)
        self.viewer_b = ImageViewer(img2, 'B', self.point_selected_test)

        viewer_layout.addWidget(self.viewer_a)
        viewer_layout.addWidget(self.viewer_b)

        # Controls
        controls_layout = QHBoxLayout()
        
        self.crop_size_spin = QSpinBox()
        self.crop_size_spin.setRange(50, 500)
        self.crop_size_spin.setValue(CROP_SIZE)
        controls_layout.addWidget(QLabel("Crop Size:"))
        controls_layout.addWidget(self.crop_size_spin)
        
        reset_btn = QPushButton("Reset Points")
        reset_btn.clicked.connect(self.reset_points_test)
        controls_layout.addWidget(reset_btn)
        
        self.test_counter_label = QLabel(f"Pairs saved: {self.save_counter - 1}")
        controls_layout.addWidget(self.test_counter_label)
        
        controls_layout.addStretch()

        main_layout.addLayout(viewer_layout)
        main_layout.addLayout(controls_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def init_evaluation_mode(self):
        """Initialize evaluation mode (1 image input)"""
        self.setWindowTitle("Dataset Generator - Evaluation Mode (1 Image)")
        self.setGeometry(100, 100, 1200, 800)

        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.jpeg)")
        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            if len(files) == 1:
                self.image_path = files[0]
                self.init_evaluation_ui()
            else:
                print("Please select exactly 1 image.")
                sys.exit(1)
        else:
            sys.exit(1)

    def init_evaluation_ui(self):
        """Create UI for evaluation mode"""
        main_layout = QHBoxLayout()
        
        # Left side: Image viewer
        left_layout = QVBoxLayout()
        
        # Image viewer
        self.viewer = ImageViewer(self.image_path, 'EVAL', self.point_selected_eval)
        left_layout.addWidget(QLabel("Click on the image to generate crop pairs:"))
        left_layout.addWidget(self.viewer)
        
        # Right side: Controls
        right_layout = QVBoxLayout()
        
        # Parameters group
        params_group = QGroupBox("Generation Parameters")
        params_layout = QFormLayout()
        
        self.crop_size_spin = QSpinBox()
        self.crop_size_spin.setRange(50, 500)
        self.crop_size_spin.setValue(CROP_SIZE)
        params_layout.addRow("Crop Size (px):", self.crop_size_spin)
        
        self.max_translation_spin = QSpinBox()
        self.max_translation_spin.setRange(0, 100)
        self.max_translation_spin.setValue(MAX_TRANSLATION)
        params_layout.addRow("Max Translation (px):", self.max_translation_spin)
        
        self.max_rotation_spin = QSpinBox()
        self.max_rotation_spin.setRange(0, 180)
        self.max_rotation_spin.setValue(MAX_ROTATION)
        params_layout.addRow("Max Rotation (deg):", self.max_rotation_spin)
        
        self.enable_rotation_cb = QCheckBox()
        self.enable_rotation_cb.setChecked(True)
        params_layout.addRow("Enable Rotation:", self.enable_rotation_cb)
        
        self.enable_translation_cb = QCheckBox()
        self.enable_translation_cb.setChecked(True)
        params_layout.addRow("Enable Translation:", self.enable_translation_cb)
        
        params_group.setLayout(params_layout)
        right_layout.addWidget(params_group)
        
        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.counter_label = QLabel(f"Pairs generated: {self.save_counter - 1}")
        self.last_transform_label = QLabel("Last transformation: None")
        
        status_layout.addWidget(self.counter_label)
        status_layout.addWidget(self.last_transform_label)
        
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)
        
        # Buttons
        reset_btn = QPushButton("Reset Counter")
        reset_btn.clicked.connect(self.reset_counter_eval)
        right_layout.addWidget(reset_btn)
        
        export_gt_btn = QPushButton("Export Ground Truth JSON")
        export_gt_btn.clicked.connect(self.export_ground_truth)
        right_layout.addWidget(export_gt_btn)
        
        open_folder_btn = QPushButton("Open Output Folder")
        open_folder_btn.clicked.connect(self.open_output_folder)
        right_layout.addWidget(open_folder_btn)
        
        # Instructions
        instructions = QLabel("""
Instructions:
1. Click anywhere on the image
2. Original crop will be saved as 'a_XXX.png'
3. Transformed crop will be saved as 'b_XXX.png' (clipped from transformed location)
4. Adjust parameters as needed
5. Continue clicking to generate more pairs
6. Export ground truth JSON when finished

Transformation details:
- Random translation: ±X pixels in x,y
- Random rotation: ±Y degrees
- b_XXX.png is clipped from the transformed location on original image
        """)
        instructions.setWordWrap(True)
        instructions.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; }")
        right_layout.addWidget(instructions)
        
        right_layout.addStretch()
        
        # Combine layouts
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setMaximumWidth(300)
        
        main_layout.addWidget(left_widget, 3)  # 3/4 of space
        main_layout.addWidget(right_widget, 1)  # 1/4 of space
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # Test mode methods
    def point_selected_test(self, label, point):
        """Handle point selection in test mode"""
        if label == 'A':
            self.point_a = point
        elif label == 'B':
            self.point_b = point

        if self.point_a and self.point_b:
            self.save_crops_test()
            self.point_a = None
            self.point_b = None
            self.viewer_a.selected_point = None
            self.viewer_b.selected_point = None

    def save_crops_test(self):
        """Save crops in test mode"""
        crop_size = self.crop_size_spin.value()
        half = crop_size / 2

        # Create folders
        os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)

        # Crop Image A
        xA, yA = self.point_a
        rectA = QRectF(xA - half, yA - half, crop_size, crop_size).intersected(QRectF(0, 0, self.viewer_a.pixmap.width(), self.viewer_a.pixmap.height()))
        if not rectA.isEmpty():
            croppedA = self.viewer_a.pixmap.copy(rectA.toRect())
            pathA = os.path.join(OUTPUT_DIR_TEST, f"{IMG_A}_{self.save_counter:03d}.png")
            croppedA.save(pathA)
            print(f"Saved: {pathA}")
        else:
            print("Point A out of bounds")

        # Crop Image B
        xB, yB = self.point_b
        rectB = QRectF(xB - half, yB - half, crop_size, crop_size).intersected(QRectF(0, 0, self.viewer_b.pixmap.width(), self.viewer_b.pixmap.height()))
        if not rectB.isEmpty():
            croppedB = self.viewer_b.pixmap.copy(rectB.toRect())
            pathB = os.path.join(OUTPUT_DIR_TEST, f"{IMG_B}_{self.save_counter:03d}.png")
            croppedB.save(pathB)
            print(f"Saved: {pathB}")
        else:
            print("Point B out of bounds")

        self.save_counter += 1
        self.test_counter_label.setText(f"Pairs saved: {self.save_counter - 1}")

    def reset_points_test(self):
        """Reset points in test mode"""
        print("Reset points")
        self.point_a = None
        self.point_b = None
        self.viewer_a.selected_point = None
        self.viewer_b.selected_point = None

    # Evaluation mode methods
    def point_selected_eval(self, label, point):
        """Handle point selection in evaluation mode"""
        crop_size = self.crop_size_spin.value()
        max_translation = self.max_translation_spin.value()
        max_rotation = self.max_rotation_spin.value()
        enable_rotation = self.enable_rotation_cb.isChecked()
        enable_translation = self.enable_translation_cb.isChecked()
        
        self.save_crop_pair(point, crop_size, max_translation, max_rotation, 
                           enable_translation, enable_rotation)

    def save_crop_pair(self, center_point, crop_size, max_translation, max_rotation,
                      enable_translation, enable_rotation):
        """Generate and save original and transformed crop pair"""
        x_center, y_center = center_point
        half_size = crop_size / 2
        
        # Create output directory
        os.makedirs(OUTPUT_DIR_EVAL, exist_ok=True)
        
        # Generate random transformation parameters FIRST
        transform_info = []
        
        # Random translation
        if enable_translation and max_translation > 0:
            tx = random.uniform(-max_translation, max_translation)
            ty = random.uniform(-max_translation, max_translation)
            transform_info.append(f"Translation: ({tx:.1f}, {ty:.1f})")
        else:
            tx, ty = 0, 0
        
        # Random rotation
        if enable_rotation and max_rotation > 0:
            rotation_angle = random.uniform(-max_rotation, max_rotation)
            transform_info.append(f"Rotation: {rotation_angle:.1f}°")
        else:
            rotation_angle = 0
        
        # Calculate transformed center point
        transformed_x = x_center + tx
        transformed_y = y_center + ty
        
        # Check if both crops are within image bounds
        img_rect = QRectF(0, 0, self.viewer.pixmap.width(), self.viewer.pixmap.height())
        
        # Original crop bounds
        crop_rect_a = QRectF(x_center - half_size, y_center - half_size, crop_size, crop_size)
        if not img_rect.contains(crop_rect_a):
            print(f"Warning: Original crop at ({x_center:.0f}, {y_center:.0f}) extends beyond image bounds")
            crop_rect_a = crop_rect_a.intersected(img_rect)
            if crop_rect_a.isEmpty():
                print("Error: No valid original crop area")
                return
        
        # Transformed crop bounds (without rotation for simple bounds check)
        crop_rect_b = QRectF(transformed_x - half_size, transformed_y - half_size, crop_size, crop_size)
        if not img_rect.contains(crop_rect_b):
            print(f"Warning: Transformed crop at ({transformed_x:.0f}, {transformed_y:.0f}) extends beyond image bounds")
            # For rotated crops, we need a larger safe area
            safe_margin = half_size * 1.5  # Extra margin for rotation
            safe_rect = QRectF(transformed_x - safe_margin, transformed_y - safe_margin, 
                              safe_margin * 2, safe_margin * 2)
            if not img_rect.contains(safe_rect):
                print("Error: Transformed crop area (with rotation) extends beyond image bounds")
                return
        
        # Save original crop (a_XXX.png)
        original_crop = self.viewer.pixmap.copy(crop_rect_a.toRect())
        original_path = os.path.join(OUTPUT_DIR_EVAL, f"{IMG_A}_{self.save_counter:03d}.png")
        original_crop.save(original_path)
        print(f"Saved original: {original_path}")
        
        # Create transformed crop (b_XXX.png) by clipping from transformed location
        transformed_crop = self.create_transformed_location_crop(
            transformed_x, transformed_y, rotation_angle, crop_size
        )
        
        # Save transformed crop
        transformed_path = os.path.join(OUTPUT_DIR_EVAL, f"{IMG_B}_{self.save_counter:03d}.png")
        transformed_crop.save(transformed_path)
        print(f"Saved transformed: {transformed_path}")
        
        # Store ground truth data
        ground_truth_entry = {
            "pair_id": self.save_counter,
            "image_a": f"{IMG_A}_{self.save_counter:03d}.png",
            "image_b": f"{IMG_B}_{self.save_counter:03d}.png",
            "original_center": [float(x_center), float(y_center)],
            "transformed_center": [float(transformed_x), float(transformed_y)],
            "translation": [float(tx), float(ty)],
            "rotation_degrees": float(rotation_angle),
            "crop_size": crop_size,
            "transformations_enabled": {
                "translation": enable_translation,
                "rotation": enable_rotation
            }
        }
        self.ground_truth_data.append(ground_truth_entry)
        
        # Update UI
        self.counter_label.setText(f"Pairs generated: {self.save_counter}")
        transform_text = "; ".join(transform_info) if transform_info else "No transformation"
        self.last_transform_label.setText(f"Last transformation: {transform_text}")
        
        self.save_counter += 1

    def create_transformed_location_crop(self, center_x, center_y, rotation_angle, crop_size):
        """Create crop from the original image at the transformed location with rotation"""
        half_size = crop_size / 2
        
        if rotation_angle == 0:
            # Simple case: no rotation, just crop at translated position
            crop_rect = QRectF(center_x - half_size, center_y - half_size, crop_size, crop_size)
            return self.viewer.pixmap.copy(crop_rect.toRect())
        
        # For rotation: create a larger extraction area and then rotate and crop
        # Calculate the diagonal of the crop to ensure we capture enough area after rotation
        diagonal = crop_size * math.sqrt(2)
        extraction_size = int(diagonal * 1.2)  # 20% margin for safety
        extraction_half = extraction_size / 2
        
        # Extract larger area from original image
        extraction_rect = QRectF(center_x - extraction_half, center_y - extraction_half, 
                               extraction_size, extraction_size)
        
        # Ensure extraction area is within image bounds
        img_rect = QRectF(0, 0, self.viewer.pixmap.width(), self.viewer.pixmap.height())
        if not img_rect.contains(extraction_rect):
            # If extraction area goes beyond bounds, we need to handle this carefully
            print(f"Warning: Extraction area for rotation extends beyond image bounds")
            extraction_rect = extraction_rect.intersected(img_rect)
        
        large_crop = self.viewer.pixmap.copy(extraction_rect.toRect())
        
        # Create canvas for rotation
        canvas = QPixmap(extraction_size, extraction_size)
        canvas.fill(Qt.black)
        
        # Draw the extracted area onto canvas (centered)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Calculate offset to center the extracted area (convert to int)
        x_offset = int((extraction_size - large_crop.width()) / 2)
        y_offset = int((extraction_size - large_crop.height()) / 2)
        painter.drawPixmap(x_offset, y_offset, large_crop)
        painter.end()
        
        # Now rotate the canvas around its center
        rotated_canvas = QPixmap(extraction_size, extraction_size)
        rotated_canvas.fill(Qt.black)
        
        painter = QPainter(rotated_canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Rotate around center
        painter.translate(extraction_size / 2, extraction_size / 2)
        painter.rotate(rotation_angle)
        painter.translate(-extraction_size / 2, -extraction_size / 2)
        painter.drawPixmap(0, 0, canvas)
        painter.end()
        
        # Extract final crop from center of rotated canvas
        final_x = (extraction_size - crop_size) / 2
        final_y = (extraction_size - crop_size) / 2
        final_rect = QRectF(final_x, final_y, crop_size, crop_size)
        
        return rotated_canvas.copy(final_rect.toRect())

    def reset_counter_eval(self):
        """Reset the save counter in evaluation mode"""
        self.save_counter = 1
        self.ground_truth_data = []  # Clear ground truth data
        self.counter_label.setText(f"Pairs generated: 0")
        self.last_transform_label.setText("Last transformation: None")
        print("Counter and ground truth data reset")

    def export_ground_truth(self):
        """Export ground truth data to JSON file"""
        if not self.ground_truth_data:
            print("No ground truth data to export")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR_EVAL, exist_ok=True)
        
        # Prepare data with metadata
        export_data = {
            "metadata": {
                "dataset_type": "controlled_evaluation",
                "total_pairs": len(self.ground_truth_data),
                "generation_parameters": {
                    "max_translation": self.max_translation_spin.value(),
                    "max_rotation": self.max_rotation_spin.value(),
                    "default_crop_size": self.crop_size_spin.value()
                }
            },
            "ground_truth": self.ground_truth_data
        }
        
        # Save to JSON file
        json_path = os.path.join(OUTPUT_DIR_EVAL, "ground_truth.json")
        try:
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Ground truth data exported to: {json_path}")
            print(f"Total pairs: {len(self.ground_truth_data)}")
        except Exception as e:
            print(f"Error exporting ground truth: {e}")

    def open_output_folder(self):
        """Open the output folder in file explorer"""
        import subprocess
        import platform
        
        output_dir = OUTPUT_DIR_EVAL if self.mode == 1 else OUTPUT_DIR_TEST
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if platform.system() == "Windows":
            os.startfile(output_dir)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", output_dir])
        else:  # Linux and others
            subprocess.Popen(["xdg-open", output_dir])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())