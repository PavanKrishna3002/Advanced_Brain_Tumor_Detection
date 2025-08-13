import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template, redirect, url_for ,send_from_directory
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np 
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Define the directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model (assuming it is already trained)
class VGG19(nn.Module):
    def __init__(self, num_classes=4):  # Adjust the number of classes accordingly
        super(VGG19, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool1(x)

        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool2(x)

        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool3(x)

        # Block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool4(x)

        # Block 5
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool5(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Initialize the model and load weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG19(num_classes=4)  # Adjust the number of classes as necessary
model.load_state_dict(torch.load('modelv19.h5', map_location=device))
model.to(device)
model.eval()


# Route for home page (index)
@app.route('/')
def index():
    return render_template('index.html')

## Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the image
        try:
            image = cv2.imread(file_path)
            if image is not None:
                image_1 = cv2.resize(image, (224, 224))
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")

        image = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Perform prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        # Map predicted class index to human-readable label
        labels = ['GLIOMA', 'MENINGIOMA', 'NO TUMOUR', 'PITUITARY']
        prediction = labels[predicted.item()]
        print(prediction)

        # Generate the URL to the uploaded image
        image_url = url_for('uploaded_file', filename=filename)

        # Pass the relevant data to the template
        return render_template('result.html', 
                               name=request.form['name'], 
                               age=request.form['age'], 
                               gender=request.form['gender'], 
                               image_url=image_url,  # Use the generated URL
                               result=prediction)
    return redirect(request.url)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# Check if the file is allowed (validate file extension)
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
