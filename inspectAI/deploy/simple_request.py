# USAGE
# python simple_request.py
# Cycle through all images in data folder and time it.
# import the necessary packages
import datetime
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
print(datetime.datetime.now())
# IP address need to set to host IP
KERAS_REST_API_URL = "http://34.216.176.105:5000/predict"
for i in range(1336):
        IMAGE_PATH = '../../data/crop_mm/' + 'IMG' + str(i) + '.jpg'

        # load the input image and construct the payload for the request
        image = open(IMAGE_PATH, "rb").read()
        payload = {"image": image}

        # submit the request
        r = requests.post(KERAS_REST_API_URL, files=payload).json()

        # ensure the request was sucessful
        if r["success"]:
                # loop over the predictions and display them
                for (i, result) in enumerate(r["predictions"]):
                        print("{}. {}: {:.4f}".format(i + 1, result["label"],
                                result["probability"]))

        # otherwise, the request failed
        else:
                print("Request failed")
print(datetime.datetime.now())
