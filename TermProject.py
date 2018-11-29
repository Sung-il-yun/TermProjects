from flask import Flask, render_template,request,redirect
from flask_uploads import UploadSet, configure_uploads, IMAGES
from werkzeug import secure_filename
import os
from flask_cors import CORS
from flask import jsonify
from InstagramAPI import InstagramAPI
import urllib.request

app = Flask(__name__, template_folder='templates')
CORS(app)

@app.route("/final")
def final():
    return render_template("Final.html")

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if request.method == 'POST':
        if 'imfile' not in request.files:
            print('NO FILE PART')
            return redirect(request.url)
        imfile = request.files['imfile']
        if imfile.filename == '':
            print('NO SELECTED FILE')
            response = {"error":"true"}
            return jsonify(response)
        if imfile:
            filepath = os.path.join("static",imfile.filename)
            imfile.save(filepath)

            response = {"error":"false", "upload_url":filepath}
            return redirect("//127.0.0.1:10000/final")

        response = {"error":"true"}
        return jsonify(response)

@app.route('/')
def index():
    return render_template("TermProject.html")


if __name__ == '__main__':
    app.run(debug = True, host="0.0.0.0", port=int(os.getenv('VCAP_APP_PORT','10000')))