from flask import Flask, render_template
import os
from flask_cors import CORS
from flask import jsonify
from instagram.client import InstagramAPI
import urllib.request

app = Flask(__name__, template_folder='templates')
app.secret_key = os.environ.get('SECERT_KEY')

instagramConfig = {
    'client_id':os.environ.get('CLIENT_ID'),
    'client_secret':os.environ.get('CLIENT_SECRET'),
    'redirect_uri':os.environ.get('REDIRECT_URI')
}
api = InstagramAPI(**instagramConfig)

@app.route("/userPhoto")
def userPhoto():
    if instagram_access_token in session and 'instagram_user' in session:
        userAPI = InstagramAPI(access_token=session['instagram_access_token'])
        recent_media, next = userAPI.user_recent_media(user_id=session['instagram_user'].get('id'),count=25)
        templateData = {
            'size' : request.args.get('size','thumb'),
            'media': recent_media
        }
        return render_template('display.html',**templateData)
    else:
        return redirect('/connect')
@app.route('/connect')
def main():
    url = api.get_authorize_url(scope=["likes", "comments"])
    return redirect(url)
@app.route("/") #웹 브라우저에서 127.0.0.1:10000/, 즉 메인 페이지 접속 시 아래 함수를 실행한다.
def index():
    return render_template("TermProject.html")




if __name__ == '__main__':
    app.run(debug = True, host="0.0.0.0", port=int(os.getenv('VCAP_APP_PORT','10000')))
