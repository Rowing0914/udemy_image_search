from src.img_saver import img_saver
from src.my_script import hello
from flask import Flask, render_template

app = Flask(__name__, static_url_path='/static')
ims = img_saver()

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/hello')
def helloWorld():
	return hello()

@app.route('/trainAgain')
def trainAgain():
	ims.main()
	return render_template('index.html')

if __name__ == '__main__':
	app.run(port=5050, host='0.0.0.0')