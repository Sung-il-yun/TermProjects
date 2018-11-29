from flask import Flask, render_template,request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from werkzeug import secure_filename
import os
from flask_cors import CORS
from flask import jsonify
from InstagramAPI import InstagramAPI
import urllib.request

app = Flask(__name__)

@app.route["upload_image", methods=["POST"]]
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
            return jsonify(response)

        response={"error":"true"}
        return jsonify(response)


@app.route('/upload')
def render_file():
    return render_template('Termproject.html')

@app.route('/fileUpload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        p = request.files['file']
        p.save(secure_filename(p.filename))
        return 'uploads 디렉토리 -> 파일 업로드 성공'


if __name__ == '__main__':
    app.run(debug = True, host="0.0.0.0", port=int(os.getenv('VCAP_APP_PORT','10000')))