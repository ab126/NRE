import sys, win32api
from flask import Flask, render_template

data_to_pass_back = 'Send this to node process'

app = Flask(__name__)


@app.route('/')
def index():
    win32api.MessageBox(0, 'Some text', 'Another text', 0x00001000)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
