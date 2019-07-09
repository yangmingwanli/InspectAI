# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@<image_file> 'http://localhost (or host IP):5000/predict'
# Submita requests via Python:
#	python simple_request.py

# import the necessary packages
from keras.models import load_model
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_kera_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = load_model('../train/mm_scratch.h5')
	global graph
	graph = tf.get_default_graph()

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = image / 255.0
	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(160, 160))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			with graph.as_default():
				preds = model.predict(image)
				preds = np.squeeze(preds)
			# results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []
			print(preds)
			# loop over the results and add them to the list of
			# returned predictions
			for (label, prob) in zip(['partialm', 'offcenterm', 'badshape', 'badsurface'], preds):
				print(label,prob)
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_kera_model()
	app.run(host='0.0.0.0', port=5000, debug=False)
