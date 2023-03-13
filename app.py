from torch_utils import NeuralNet

import io
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])

# Define the model
Input_size = 784 # 28x28
Hidden_size = 500 
Num_classes = 10
model = NeuralNet(Input_size, Hidden_size, Num_classes)
model.load_state_dict(torch.load('mnist_ffn.pth'))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Define a function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_text = ''
    # Check if an image was uploaded
    if 'file' not in request.files:
        return jsonify({'prediction': 'No file uploaded'})
    
    file = request.files['file']
    
    # Check if the file extension is allowed
    if not allowed_file(file.filename):
        return jsonify({'prediction': 'Invalid file type'})
    
    # Save the image to disk
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    # Load the image and apply the transformation
    image = Image.open(filename)
    image_tensor = transform(image).view(1,-1)
    
    # Make a prediction using the model
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        prediction = predicted.item()

    prediction = str(prediction)

    # Return the predicted result
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)