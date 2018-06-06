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
iterations = int(arguments[4])
batchSize = int(arguments[5])

classes,dataX,dataYa,dataY = util.processAudioForTest(bpm,samplingRate,mypath)

def test():

	myModel = SoundCNN(classes)
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		saver = tf.train.Saver(tf.all_variables())
		save_path = "./model.ckpt"
		saver.restore(sess, save_path)

		print "Feature vector for each data"
		print(dataX)
		print("# Data: %d", len(dataX))

		'''
		Class label: 0 -> Class vector: [1, 0, ..., 0]
		Class label: 1 -> Class vector: [0, 1, ..., 0]
		'''
		print "Class label for each data"
		print(dataYa)
		print "Class vector for each data"
		print(dataY)

		# Prediction
		predictions = sess.run(myModel.y_conv, feed_dict={myModel.x: dataX, myModel.keep_prob: 1.0})
		print "PREDICTION RESULT (Class vector for each data)"
		print predictions
	
		# Calculate accuracy
		test_accuracy = sess.run(myModel.accuracy, feed_dict={myModel.x: dataX, myModel.y_: dataY, myModel.keep_prob: 1.0})
		print "PREDICTION ACCURACY"
		print test_accuracy

test()

