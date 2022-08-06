# import required packages
import glob
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence

from utils import *
# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


# -------------------------------------------------------------------------------------------------
#           Class for Natural Language Processing (NLP)
# -------------------------------------------------------------------------------------------------
class NLP ():
	# ---- Set-up and Compile the network ----
	def __init__ (self, no_feat, loss_function, optimizer, batch_size):
		self.batch_size = batch_size

		self.model = Sequential([
				# Embedding layer with 'No. of unique features + 1' as the input dimention, and 32 as the output dimension
				# Refer report for reason behind its use
				layers.Embedding (no_feat + 1, 64),
				
				# 2 Convolution Layers - each with 32 filters and kernels of size 4 and 3 respectively
				# Refer report for reason behind their use
				layers.Conv1D(filters=64, padding = "valid", activation="relu", kernel_size = 4),	# Convolution Layer - 1
				layers.Conv1D(filters=32, padding = "valid", activation="relu", kernel_size = 3),	# Convolution Layer - 2
				layers.GlobalMaxPooling1D(),            # Apply GlobalMaxPooling after the Covolution Layer; also converts it to a compatible form going further
				
				layers.Dense(64, activation="relu"),   	# Dense Layer with Relu, to help the model capture complex relationships
				layers.Dropout(rate = 0.1),				# Dropout layer
				layers.Dense(32, activation="relu"),	# Dense Layer with Relu again
				layers.Dropout(rate = 0.1),				# Dropout layer
				layers.Dense(16, activation="relu"),	# Dense Layer with Relu again
				layers.Dropout(rate = 0.1),				# Dropout layer
				layers.Dense(2, activation='softmax')   # Dense Layer, acting as the output layer (using softmax - for classification)
			])

		self.model.compile(loss = loss_function, optimizer = optimizer, metrics=['accuracy'])

	# ---- Train and test the network with given ----
	def train_and_test (self, train_input, train_label, val_input, val_label, no_epochs):
		return self.model.fit(train_input, train_label, epochs = no_epochs,  batch_size = self.batch_size, validation_data=(val_input, val_label))

	# ---- Test the network with given ----
	def test (self, input, label):
		return self.model.evaluate(input, label)

	# ---- Predict Outcomes for the given Inputs ----
	def predict (self, input):
		return self.model.predict(input)
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------
#           Class with functions to prepare, preprocess and perform other operations on the data
# -------------------------------------------------------------------------- -----------------------
class Data_Store ():
	def __init__ (self, classes):
		self.classes = classes

		self.train_input = None
		self.train_label = None

		self.test_input = None
		self.test_label = None

		self.total_features = 0


	# Function to find train and test data
	def prepare_data (self, data_dir):
		self.train_input, self.train_label = self.extract_input_label(data_dir + "/train")
		self.test_input, self.test_label = self.extract_input_label(data_dir + "/test")


	# Function to extract review_content and rating(classification) from each file in Train/Test Folder
	#  - [References: https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/]
	def extract_input_label (self, data_dir):
		input = list()
		label = list()

		count = 0
		for cl in self.classes:
			scan = os.path.join(data_dir, cl, "*")
			rating = ([1, 0] if cl == "pos" else [0, 1])

			for review_file in glob.glob(scan, recursive = False):
				review_content = open(review_file, "r", encoding = "utf8").read()

				input.append(review_content)
				label.append(rating)
				
				count += 1
				# print(count)

		return input, label

	# Function to convert the input strings to vector integers
	def tokenize_input (self):
		converter = text.Tokenizer()
		converter.fit_on_texts(self.train_input)

		self.train_input, self.test_input = converter.texts_to_sequences(self.train_input), converter.texts_to_sequences(self.test_input)
		self.total_features = len(converter.word_index)

	# Function to convert the input to a uniform format
	# - We want the 2d array to have the same number of features (columns) per each input (row)
	# - Hence we zero-pad the input
	def zero_pad_input (self):
		# Decide on the padding length
		each_length = [len(row) for row in self.test_input]
		padding_length = int(max((sum(each_length) / len(self.test_input)), max(each_length) / 2))

		# Zero-pad
		self.train_input, self.test_input = sequence.pad_sequences(self.train_input, padding = "post", maxlen = padding_length), sequence.pad_sequences(self.test_input, padding = "post", maxlen = padding_length)
	
	# Function to convert the arrays/lists to numpy arrays
	def convertToNumpy (self):
		self.train_input, self.train_label = np.array(self.train_input, dtype="int"), np.array(self.train_label, dtype="int")
		self.test_input, self.test_label = np.array(self.test_input, dtype="int"), np.array(self.test_label, dtype="int")
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------







# -------------------------------------------------------------------------------------------------
#					Main Function
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	# *************************************************
	#           Main-Specific Data
	# *************************************************
	# Optimizer
	G_optimizer = 'adam'

	# Common Loss Function
	G_loss_function = 'binary_crossentropy' 	# Works well with softmax for 2 classes

	# No. of epochs to train for
	G_no_epochs = 5

	# Batch_Size
	G_batch_size = 1024

	# *************************************************
	# 0. Prepare and preprocess the data
	# *************************************************
	data = Data_Store(classes = ["pos", "neg"])

	# Read from the directory for training-data
	data.prepare_data("./data/aclImdb")

	# Convert input to corresponding integer form, that is better understandable to the machine
	data.tokenize_input()

	# Convert the input data to a uniform format, by padding each row to the maximum possible length of inputs
	data.zero_pad_input()

	# Convert the input_data into numpy format, so that it can by used well with the Neural Network
	data.convertToNumpy()

	# print(data.input.shape)
	# print(data.input[0])
	# print(data.label.shape)
	# print(data.label[0])
	
		
	# *************************************************
	#		1. load your training data
	# *************************************************
	# Split the data into train_val sets (90%-10%)
	train_input, val_input, train_label, val_label = train_test_split(data.train_input, data.train_label, test_size= 0.1, random_state=35)



	# *************************************************
	# 2. Train your network
	# Make sure to print your training loss within training to show progress 

	# Make sure you print the final training loss
	# *************************************************
	nlp = NLP(no_feat = data.total_features, loss_function = G_loss_function, optimizer = G_optimizer, batch_size=G_batch_size)
	nlp_results = nlp.train_and_test(train_input = train_input, train_label = train_label, val_input = val_input, val_label = val_label, no_epochs = G_no_epochs)

	# Draw Plots
	g = Graphs()
	g.plot(input = nlp_results, title = 'NLP Loss', plot_1 = 'loss', plot_2 = 'val_loss', x_label='Epoch', y_label='Loss')
	g.plot(input = nlp_results, title = 'NLP Accuracy', plot_1 = 'accuracy', plot_2 = 'val_accuracy', x_label='Epoch', y_label='Accuracy')

	# Report Results
	print("\n--------------------------------------------------------------------------------------")
	print("After " + str(G_no_epochs) + " Epochs")
	print("--------------------------------------------------------------------------------------")
	print("Training Loss: " + str(nlp_results.history['loss'][-1]))
	print("Validation Loss: " + str(nlp_results.history['val_loss'][-1]))
	print("Training Accuracy: " + str(nlp_results.history['accuracy'][-1] * 100) + " %")
	print("Validation Accuracy: " + str(nlp_results.history['val_accuracy'][-1] * 100) + " %")


	
	# *************************************************
	#		3. Save your model
	# *************************************************
	nlp.model.save("./models/Group52_NLP_model.h5")
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------