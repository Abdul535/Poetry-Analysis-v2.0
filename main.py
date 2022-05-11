from flask import Flask,request,render_template,redirect
import os
app = Flask(__name__)


app.config["CSV_UPLOAD"] = "./input"
#app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]

from werkzeug.utils import secure_filename

@app.route('/',methods = ["GET"])
def home():
	return render_template('main.html')


@app.route('/',methods = ["POST"])
def upload_csv():
	if request.method == "POST":
		csv = request.files['file']

		if csv.filename == '':
			print("csv must have a file name")
			return redirect(request.url)


		filename = secure_filename(csv.filename)

		basedir = os.path.abspath(os.path.dirname(__file__))
		csv.save(os.path.join(basedir,app.config["CSV_UPLOAD"],'all'))

		return render_template("main.html")



# @app.route('/')
# def display_image(filename):
# 	# return redirect(url_for('static',filename = "/Images" + filename), code=301)
# 	return render_template('main.html')


app.run(debug=True,port=5000)