import os
from flask import Flask, jsonify, request, send_from_directory, render_template
from werkzeug.utils import secure_filename
import sys
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = 'localhost:5000/'
PORT = int(os.getenv('PORT', 5000))

sys.path.append("..")
app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'result/'

class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:
                objects_contours.append(cnt)
        return objects_contours

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def upload_file_to_bucket(file, filename, dest):
    source = os.path.join(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Load Aruco detector
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            parameters = cv2.aruco.DetectorParameters()

            # Load Object Detector
            detector = HomogeneousBgDetector()

            # Load Image From Storage Bucket 
            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Get Aruco marker
            corners, _, _ = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

            # Draw polygon around the marker
            int_corners = np.int0(corners)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

            # Aruco Perimeter
            aruco_perimeter = cv2.arcLength(corners[0], True)

            # Pixel to cm ratio
            pixel_cm_ratio = aruco_perimeter / 20

            contours = detector.detect_objects(img)

            height_list = []
            width_list = []

            # Draw objects boundaries
            for i, cnt in enumerate(contours):
                # Get rect
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect

                # Get Width and Height of the Objects by applying the Ratio pixel to cm
                object_width = w / pixel_cm_ratio
                object_height = h / pixel_cm_ratio

                height_list.append(object_height)
                width_list.append(object_width)

                # Display rectangle
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.polylines(img, [box], True, (255, 0, 0), 2)
                cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

            cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], filename), img)

            publicUrl = os.path.join(ROOT_PATH, 'result', 'Result-' + filename)

            idx = np.argmax(height_list)
            response_data = {
                    'status': 'success',
                    'message': 'Prediksi Tinggi Badan Berhasil (Versi Beta)',
                    'data': {
                        'image_url': publicUrl,
                        'listHeight': height_list,
                        'listWidth': width_list,
                        'tinggiBadan': height_list[idx],
                    },
                }
            return jsonify(response_data), 200
        except IndexError as e:
            response_data = {
                'status': 'failed',
                'message': f'Error: AruCo Marker tidak terdeteksi',
                'data': [],
            }
            return jsonify(response_data), 500

    else:
        response_data = {
            'status': 'failed',
            'message': 'Silakan unggah foto anak yang jelas dengan AruCo Marker yang terlihat!',
            'data': [],
        }
        return jsonify(response_data), 400

@app.route('/result/<name>')
def output_file(name):
    return send_from_directory(os.path.abspath(app.config['OUTPUT_FOLDER']), name)

@app.errorhandler(404)
def not_found(error):
    response_data = {'status': 'failed', 'message': 'Endpoint not found', 'data': [], 'status_code': 404}
    return jsonify(response_data), 404

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=PORT)