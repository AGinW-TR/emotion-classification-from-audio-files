from flask import Flask, request, render_template
import os
import shutil
import tempfile
import time
# from your_model_file import predict_emotion  # Import your model's prediction function

app = Flask(__name__)

# Ensure there's a folder for uploaded files
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    emotion = None  # Default to None if no file has been processed
    plot_name = None
    audio_name = None

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('upload.html', message='No file part', emotion=emotion)
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', message='No selected file', emotion=emotion)
        if file:
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))
            shutil.copy(os.path.join(UPLOAD_FOLDER, file.filename), 
                        os.path.join('static', file.filename))
            
            # Predict emotion
            time.sleep(1)
            # emotion = 'sample output 2'
            plot_name = file.filename.split('.')[0] + '.jpg'
            audio_name = file.filename

            # emotion = predict_emotion(os.path.join(UPLOAD_FOLDER, file.filename))  # Implement this function based on your model

    return render_template('upload.html', emotion=emotion, plot_name=plot_name, audio_name=audio_name)


if __name__ == '__main__':
    app.run(debug=True)
