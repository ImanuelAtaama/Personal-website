from flask import Flask, render_template, request, redirect, url_for
import os
import cloudinary
import cloudinary.uploader
from werkzeug.utils import secure_filename
from model import load_model, predict_image

app = Flask(__name__)

# Konfigurasi Cloudinary (Ganti dengan kredensial akun Cloudinary Anda)
cloudinary.config(
    cloud_name= 'dqj4dyfne', 
    api_key= '674711576519494', 
    api_secret= '6YIquAcxY0fIvcosGW04y2QZG3o'
)

# Load model DenseNet121
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image_inference', methods=['GET', 'POST'])
def image_inference():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('image.html')
        file = request.files['file']
        if file.filename == '':
            return render_template('image.html')
        if file:
            filename = secure_filename(file.filename)
            if not filename.lower().endswith('.jpg'):
                return render_template('image.html', error="File harus memiliki ekstensi .jpg")
            
            # Upload ke Cloudinary
            result = cloudinary.uploader.upload(file)
            file_url = result["secure_url"]  # URL gambar setelah diunggah ke Cloudinary
            
            return redirect(url_for('result', file_url=file_url))
    return render_template('image.html')

@app.route('/result')
def result():
    file_url = request.args.get('file_url')
    label = predict_image(model, file_url)  # Kirim URL ke model untuk prediksi
    return render_template('result.html', file_url=file_url, label=label)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
