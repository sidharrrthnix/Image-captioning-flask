from flask import Flask, render_template, url_for
import os
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/temp/'
app.config['MAX_CONTENT_PATH'] = 200 * 1024 * 1024

if __name__ == '__main__':
    app.run(debug=True)