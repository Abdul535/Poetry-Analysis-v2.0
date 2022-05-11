from flask import Flask,request,render_template,redirect
import os
import time
import urllib.parse
app = Flask(__name__)


app.config["CSV_UPLOAD"] = "./input"
#app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]

from werkzeug.utils import secure_filename




@app.route('/',methods = ["GET","POST"])
def upload_csv():
	if request.method == "POST":
		csv = request.files['file']

		if csv.filename == '':
			print("csv must have a file name")
			return redirect(request.url)


		filename = secure_filename(csv.filename)
		print('input file is', filename)
	
		basedir = os.path.abspath(os.path.dirname(__file__))
		csv.save(os.path.join(basedir,app.config["CSV_UPLOAD"],'all.csv'))
		print('file saved')

		
		return render_template('main.html')
	return render_template('main.html')

@app.route('/result',methods = ["POST"])
def showres():
		if request.method == "POST":
			os.system('python analysis.py')

			filenames = ['pic1.png', 'pic2.png','pic3.png', 'pic4.png']
		
			return render_template('result.html', filenames=filenames)
		return render_template('main.html')




def convert(input):
    # Converts unicode to string
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, str):
        return input.encode('utf-8')
    else:
        return input

@app.route("/target_endpoint",methods = ["POST"])
def target():
	# You could do any information passing here if you want (i.e Post or Get request)
	some_data = "Here's some example data"
	some_data = urllib.parse.quote(convert(some_data)) # urllib2 is used if you have fancy characters in your data like "+"," ", or "="
	# This is where the loading screen will be.
	# ( You don't have to pass data if you want, but if you do, make sure you have a matching variable in the html i.e {{my_data}} )
	return render_template('loading.html', my_data = some_data)

@app.route("/processing")
def processing():
	# This is where the time-consuming process can be.
	data = "No data was passed"
	# In this case, the data was passed as a get request as you can see at the bottom of the loading.html file
	if request.args.to_dict(flat=False)['data'][0]:
		data = str(request.args.to_dict(flat=False)['data'][0])
	# This is where your time-consuming stuff can take place (sql queries, checking other apis, etc)
	# time.sleep(120) # To simulate something time-consuming, I've tested up to 100 seconds
	os.system('python analysis.py')	
	# You can return a success/fail page here or whatever logic you want

	filenames = ['pic1.png', 'pic2.png','pic3.png', 'pic4.png']		
	return render_template('result.html', filenames=filenames)

	# return render_template('success.html', passed_data = data)

		







# @app.route('/')
# def display_image(filename):
# 	# return redirect(url_for('static',filename = "/Images" + filename), code=301)
# 	return render_template('main.html')


app.run(debug=True,port=5000)