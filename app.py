import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import load_model

class SteatosisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Steatosis Level Predictor")
        self.setGeometry(100, 100, 400, 400)

        self.model = load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.image_label = QLabel("No image loaded")
        self.image_label.setScaledContents(True)
        self.result_label = QLabel("")
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(224, 224))
            self.predict_steatosis(file_path)

    def predict_steatosis(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)  # add batch dim
        with torch.no_grad():
            output = self.model(tensor)
            result = output.item()
            self.result_label.setText(f"Steatosis level: {result:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SteatosisApp()
    window.show()
    sys.exit(app.exec_())
