# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from model import load_model, predict_image

app = Flask(__name__)

# Folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('result', filename=filename))
    return render_template('image.html')

@app.route('/result/<filename>')
def result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    label= predict_image(model, filepath)
    return render_template('result.html', filename=filename, label=label)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
