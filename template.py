from flask import Flask, render_template, url_for, request
from datetime import datetime
from vocabulary import Vocabulary
from Executor import CNN_PREDICT
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)

CWD = os.getcwd()
print('Current Working Directory : '+CWD)



@app.route('/')
@app.route('/image-caption')
def home():
    """Renders the home page."""
    
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      filename = secure_filename(f.filename)
      f.save('static/temp/'+filename)
      print(filename)
      ENGLISH,GERMAN = CNN_PREDICT(CWD+'/static/temp/'+filename)
      GERMAN = GERMAN.replace('<unk>','')
      return render_template(
        'index.html',
        title='Contact',
        year=datetime.now().year,
        english=ENGLISH,
        german=GERMAN,
        filename='/static/temp/'+filename
        )
     
@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message=''
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About MID',
        year=datetime.now().year,
        message='asdad'
    )

if __name__ == '__main__':
    app.run(debug=True)