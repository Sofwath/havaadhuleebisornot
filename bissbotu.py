import tensorflow as tf
import os
from PIL import Image
import numpy as np
import cv2
import random
from io import BytesIO
import requests
import tweepy
from PIL import ImageFile
import argparse
import io
import sys

from secrets import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
auth = tweepy.OAuthHandler(C_KEY, C_SECRET)
auth.set_access_token(A_TOKEN, A_TOKEN_SECRET)
# Construct the API instance
api = tweepy.API(auth)  # create an API object

graph_def = tf.GraphDef()
labels = []
filename = 'model.pb'
labels_filename ='labels.txt'


def isHLB(myimage):
	# Load from a file
	imageFile = myimage
	image = Image.open(imageFile)
	image = update_orientation(image)
	image = convert_to_opencv(image)
	image = resize_down_to_1600_max_dim(image)
	h, w = image.shape[:2]
	min_dim = min(w,h)
	max_square_image = crop_center(image, min_dim, min_dim)
	augmented_image = resize_to_256_square(max_square_image)
	network_input_size = 227
	augmented_image = crop_center(augmented_image, network_input_size, network_input_size)
	output_layer = 'loss:0'
	input_node = 'Placeholder:0'
	with tf.Session() as sess:
	    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
	    predictions, = sess.run(prob_tensor, {input_node: [augmented_image] })
	highest_probability_index = np.argmax(predictions)
	print('Classified as: ' + labels[highest_probability_index])
	print()
	label_index = 0
	for p in predictions:
	    truncated_probablity = np.float64(round(p,8))
	    print (labels[label_index], truncated_probablity)
	    label_index += 1
	print ('------------')
	return (truncated_probablity)

def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def tweet_image(url, username, status_id):
    filename = 'temp.png'
    status = 'ތީ ހަވާދުލީބިހެއް ނޫން'
    # send a get request
    request = requests.get(url, stream=True)
    if request.status_code == 200:
        # read data from downloaded bytes and returns a PIL.Image.Image object
        i = Image.open(BytesIO(request.content))
        i.save(filename)
        biss = isHLB(filename)
        if (biss >= 1.0):
        	status = 'ތީ ހަވާދުލީބިސް'
        if (biss <= 0.9):
        	status = 'ތި ފަހަރުގަ ވެދާނެ ހަވާދުލީބިހާ'
        if (biss < 0.8):
        	status = 'ތީ ހަވާދުލީބިހެއް ނޫން'
        api.update_status(status='@{0} {1} #HavaadhuleeBis'.format(username,status), in_reply_to_status_id=status_id)
    else:
        print("unable to download image")

# create a class inheriting from the tweepy  StreamListener
class BotStreamer(tweepy.StreamListener):
    # Called when a new status arrives which is passed down from the on_data method of the StreamListener
    def on_status(self, status):
        username = status.user.screen_name
        status_id = status.id

        # entities provide structured data from Tweets including resolved URLs, media, hashtags and mentions without having to parse the text to extract that information
        if 'media' in status.entities:
            for image in status.entities['media']:
                tweet_image(image['media_url'], username, status_id)

# Import the TF graph
with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Create a list of labels.
with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())

myStreamListener = BotStreamer()
# Construct the Stream instance
stream = tweepy.Stream(auth, myStreamListener)
stream.filter(track=['@edhemeehun'])


