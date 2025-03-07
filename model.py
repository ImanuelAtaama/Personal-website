import os
import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import random
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def load_model():
    # Definisikan arsitektur model DenseNet121
    model = models.densenet121(pretrained=True)
    # Modify the classifier untuk 7 classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 640),
        nn.ReLU(),
        nn.Dropout(0),
        nn.Linear(640,128),
        nn.ReLU(),
        nn.Dropout(0),
        nn.Linear(128, 7)  # Adjust for your number of classes
    )
    
    # Load state_dict ke model
    model.load_state_dict(torch.load('model/all_densenet121_waste_classification_augmented.pth', map_location=torch.device('cpu')))
    model.eval()  # Pastikan model dalam mode evaluasi
    return model

def predict_image(model, image_path):
    image = Image.open(image_path)
    target_size = (512,384)
    
    original_width, original_height = image.size
    ratio = min(target_size[0] / original_width, target_size[1] / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    
    # Transformasi gambar agar sesuai dengan input model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Resize gambar
    resized_image  = image.resize((new_width, new_height), Image.LANCZOS)

    # Buat gambar baru dengan background hitam
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    new_image.paste(resized_image, ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2))
    
    # Menghitung ukuran baru menjadi 50% dari ukuran yang sudah diubah
    final_width = int(target_size[0] * 0.5)
    final_height = int(target_size[1] * 0.5)

    # Resize gambar yang sudah dipadding menjadi 50%
    final_image = new_image.resize((final_width, final_height), Image.LANCZOS)
    
    # Load gambar
    image = transform(image).unsqueeze(0)  # Tambahkan batch dimension
    
    # Prediksi dengan model
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        label_idx = torch.argmax(probabilities).item()
        # probability = probabilities[label_idx].item()
    
    # Ganti kategori label dengan nama sesuai dataset Anda
    categories = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']
    label = categories[label_idx]
    return label
