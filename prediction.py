import tensorflow as tf
import scipy.io.wavfile
import numpy as np
import matplotlib.mlab
from os import listdir
from os.path import isfile, join
import sys
import utilities as util

from model import SoundCNN

arguments = sys.argv

bpm = int(arguments[1])
samplingRate = int(arguments[2])
mypath = str(arguments[3])
classes = int(arguments[4])

dataX = util.processAudioForPrediction(bpm,samplingRate,mypath)

def prediction():

	myModel = SoundCNN(classes)
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		saver = tf.train.Saver(tf.all_variables())
		save_path = "./model.ckpt"
		saver.restore(sess, save_path)

		print "Feature vector for each data"
		print(dataX)
		print("# Data: %d", len(dataX))
        
		# Prediction
		predictions = sess.run(myModel.y_conv, feed_dict={myModel.x: dataX, myModel.keep_prob: 1.0})
		print "PREDICTION RESULT (Class vector for each data)"
		print predictions

prediction()

