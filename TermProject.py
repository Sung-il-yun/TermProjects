from flask import Flask, render_template, request, redirect
from flask_uploads import UploadSet, configure_uploads, IMAGES
from typing import Any
from werkzeug import secure_filename
import os
from flask_cors import CORS
from flask import jsonify
import vgg

import urllib.request

app = Flask(__name__, template_folder='templates')
CORS(app)


@app.route("/upload_image", methods=["POST"])
def upload_image():
    if request.method == 'POST':
        if 'selfimg' not in request.files:
            print('NO FILE PART')
            return redirect(request.url)
        selfimg = request.files['selfimg']
        if selfimg.filename == '':
            print('NO SELECTED FILE')
            response = {"error": "true"}
            return jsonify(response)
        if selfimg:
            filepath = os.path.join("static", selfimg.filename)
            selfimg.save(filepath)
            os.rename(filepath, 'static/selfimg')
            name, percent = vgg.test('static/selfimg')
            print(name, percent)

            return name, percent

        response = {"error": "true"}
        return jsonify(response)
"""
@app.route("/getName")
def getName():



@app.route("/getPercent")
def getPercent():
"""
@app.route('/', methods=["GET", "POST"])
def index():
    return render_template("TermProject.html")


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv('VCAP_APP_PORT', '10000')))
