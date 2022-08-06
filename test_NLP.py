# import required packages
import glob
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence

from utils import *
from train_NLP import *

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
	# *************************************************
	#		1. Load your saved model
	# *************************************************
	nlp_model = models.load_model("./models/Group52_NLP_model.h5")


	# *************************************************
	# 	2. Load your testing data
	# 	- Prepare and preprocess the data
	# *************************************************
	data = Data_Store (classes = ["pos", "neg"])
	# Read from the directoty for training-data
	data.prepare_data("./data/aclImdb")

	# Convert input to corresponding integer form, that is better understanble to the machine
	data.tokenize_input()

	# Convert the input data to a uniform format, by padding each row to the maximum possible length of inputs
	data.zero_pad_input()
	
	# Convert the input_data into numpy format, so that it can by used will with the Neural Network
	data.convertToNumpy()

	# print(data.input.shape)
	# print(data.input[0])
	# print(data.label.shape)
	# print(data.label[0])


	# *************************************************
	#	3. Run prediction on the test data and print the test accuracy
	# *************************************************
	nlp_results = nlp_model.evaluate(data.test_input, data.test_label)


	print("\n---------------------------------------------------------")
	print("          Results")
	print("-----------------------------------------------------------")
	print("Loss: " + str(nlp_results[0]))
	print("Accuracy: " + str(nlp_results[1] * 100) + " %")