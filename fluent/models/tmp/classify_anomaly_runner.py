# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
"""
Experiment runner for text classification, where the class labels are
determined by anomaly scores. Please note the following definitions:
- Training dataset: all the data files used for experimentally building the NLP
	system. During k-fold cross validation, the training dataset is split 
	differently for each of the k trials. The majority of the dataset is used for 
	trainging, and a small portion is held out for evaluation -- this evaluation
	data is different from the test data.
- Testing dataset: the data files held out until the NLP system is complete.
	That is, the system should never see this testing data and then go back and 
	change models/params/methods/etc. at the risk of overfitting.
- Classification and label are used interchangeably.

Each sample is a piece of text with a corresponding clasification. For each 
classification in the dataset, the model learns each sample as a sequence of
tokens (i.e. items of tokenized text). For a given test sequence, the anomaly
score for each classification is calculated. The sample is labeled to the
classification w/ the lowest anomaly score.
"""

import argparse
import csv
import os
import time

from fluent.bin.tools import getData  ## TODO: create; should return a list of dicts (or tuples b/c of training for loop), i.e. [{text:"baz",labels:["foo"]},{text:"buz",labels:["foo","bar"]}]
from fluent.encoders.cio_encoder import CioEncoder
from fluent.models.classification_model import ClassificationModel


# Number of folds for k-fold cross validation; for no split, k=None
# If running on the held-out test data, set k=None
k = 5


# def getData(dataDir, k):
# 	"""

# 	Returns a list of k dictionaries, each with keys for 'samples' and 
# 	'classification'.
# 	"""
# 	with open(os.path.join(self.dataPath, "training")) as trainingData:
# 		for i in range(k):



	return

def train(data, checkpointDir):
	"""
	Create a model for each classification category.

	Assumes...
	"""
	# Inits
	encoder = CioEncoder()
	models = {}
  n = 0

  start = time.time()
  for sample, labels in data:
  	text = encoder.tokenize(sample)
  	# Get or create a model for each classification.
  	for label in labels:  # may be multiple labels for a given sample
  		model = models.get(label)
  		if not model:
  			checkpointPath = os.path.join(checkpointDir, label)
  			model = ClassificationModel(checkpointPath=checkpointPath)  ## IMPLEMENT MODEL INIT PER THIS LINE
  			models[label] = model
  		model.resetSequence()

  		for token in text:
  			try:
  				model.feedText(encoder.encode(token), learn=True)
  			except KeyError:
  				print "Skipping '%s' because issue encoding." % token

  		n += 1
  		if n%10 == 0:
  			print "."
	print "Trained on %i phrases in %d seconds." % (n, time.time() - start)

  # Save the models.
  import pdb; pdb.set_trace()
  for model in models.values():
    model.resetSequence()
    model.save()


def test(data, checkpointDir, resultsDir):
	"""
	"""
	# Load the models
	models = {}
	for path in os.listdir(checkpointDir):
		####
		import pdb; pdb.set_trace()
		model = ClassificationModel(checkpointPath=path)
		model.load()
		models[checkpointPath] = model

	for sample, labels in data:
		# Build a list of anomaly scores for this sample.
		for model in models.values():
      model.resetSequence()
    text = encoder.tokenize(sample)
    for token in text: 






def main(args):

	# Setup directories
	root = os.path.dirname(os.path.realpath(__file__))
	dataDir = os.path.join(root, args.dataDir)
	resultsDir = os.path.join(root, args.resultsDir)

	# # Instantiate model
	# model = ClassificationModel(###)
	# trainingData = model.getText()
	
	if args.test: #######
		data = getData(os.path.join(dataDir, "test"))
		test(data)

		return
	else:
		data = getData(os.path.join(dataDir, "training"))

	with open(outputPath, 'w') as fout:
    writer = csv.writer(fout)
    writer.writerow(["predicted"], ["actual"])  ## call writer below...

		# Run k trials of different data splits for cross validation.
		for i in range(k):
			evalSet = trainingData[i*splitSize][:subsetSize]
			trainSet = trainingData[:i*subsetSize] + trainingData[(i+1)*subsetSize:]
		
			# Train a model for each classification, and save to checkpoint dir.
			print "Training for trial %i..." % i
			train(trainSet, args.checkpointDir)

			# Load models to test on evaluation dataset.
			print "Evaluating for trial %i..." % i
			test(evalSet, args.checkpointDir)

			print "Calculating intermediate results..."
			model.evaluateResults(predicted, actual)  # calculate confusion matrix, prec, recall
		
		print "Calculating cumulative results for %i trials...", % k


	# Find mean stats over k rounds


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-k", "--kFolds",
  										default=5,
  										type=int,
  										help="Number of folds for cross validation, or None.")
  parser.add_argument("--train",
                    help="Train the models, but do not run on evaluation data.",
                    default=False,
                    action="store_true")
  parser.add_argument("--evaluate",
                    help="Run the models on the evaluation data.",
                    default=False,
                    action="store_true")
  parser.add_argument("--test",
                    help="Run the models on the test data.",
                    default=False,
                    action="store_false")
  parser.add_argument("--dataDir",
                    	default="data",
                    	help="This directory holds both training and test "
                    	"datasets.")
  parser.add_argument("--resultsDir",
	                    default="results",
	                    help="This will hold the evaluation results.")
  parser.add_argument("--checkpointDir",
	                    default="checkpoint",
	                    help="This will save classification models.")
  args = parser.parse_args()
  main(args)