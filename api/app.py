from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import logging
import os



class Model:
    model = load_model('./api/alzheimer_model.keras')

    def analise(self, image_path):
        image_test = image.load_img(image_path, target_size=(176, 208))
        image_test = image.img_to_array(image_test)
        image_test /= 255
        image_test = np.expand_dims(image_test, axis = 0)
        prediction = self.model.predict(image_test)
        return prediction



app = Flask(__name__)
app.config['DEBUG'] = True  # Habilita o modo de depuração
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'api/static/upload')


@app.route('/', methods=['GET'])
def hello_world():
    try:
        return render_template('home.html')
    except Exception as e:
        app.logger.error(f"Exception occurred: {e}")
        return "An error occurred", 500



@app.route('/analise/', methods=['GET', 'POST'])
def contact():
    prediction = None
    try:
        ia = Model()
        if request.method == "POST":
            file = request.files['file']
            save_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(save_path)
            prediction = ia.analise(save_path)
            print(file.filename)
        return render_template('analise.html', prediction=prediction[0], file_path=file.filename)
    except Exception as e:
        app.logger.error(f"Exception occurred: {e}")
        return "An error occurred", 500






if __name__ == '__main__':
    app.run(port=8085, host='0.0.0.0', debug=True, threaded=True)