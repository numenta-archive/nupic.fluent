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

import cPickle as pickle
import numpy
import os
import shutil


class ClassificationModel(object):
	"""
	Base class for NLP models of classification tasks. When inheriting from this
  class please take note of which methods MUST be overridden, as documented
  below.

  The Model superclass implements:
    - evaluateTrialResults calcualtes result stats
    - evaluateResults() calculates result stats for a list of trial results
    - densifyPattern() returns a binary SDR vector for a given bitmap
    - save()/load() saves/loads a serialized model checkpoint

  Methods/properties that must be implemented by subclasses:
  	- encodePattern()
  	- trainModel()
  	- testModel()

	"""

################################################################################
# Experiment methods

	def evaluateTrialResults(self, classifications, n):  ## TODO: add precision, recall, F1 score
		"""
		Calculate statistics for the predicted classifications against the actual.

		@param classifications	(list)						Two lists: (0) predictions and (1)
																							actual classifications.
		@return									(tuple)						Returns a 2-item tuple w/ the
																							accuracy (float) and confusion
																							matrix (numpy array).
		"""
		if len(classifications[0]) != len(classifications[1]):
			raise ValueError("Classification lists must have same length.")
		actual = numpy.array(classifications[1])
		predicted = numpy.array(classifications[0])
		
		accuracy = (actual == predicted).sum() / float(len(actual))

		cm = numpy.zeros((n, n))
		for a, p in zip(actual, predicted):
			cm[a][p] += 1

		return (accuracy, cm)


	def evaluateResults(self, intermResults):
		"""
		Cumulative statistics for the outputs of evaluateTrialResults().

		@param intermResults			(list)					List of returned results from
																							evaluateTrialResults().
		@return										(list)					Returns a dictionary with entries
																							for max, mean, and min accuracies,
																							and the mean confusion matrix.
		"""
		accuracy = []
		cm = numpy.zeros((len(intermResults[0][1]), len(intermResults[0][1])))

		# Find mean, max, and min values for the metrics.
		k = 0
		for result in intermResults:
			accuracy.append(result[0])
			cm = numpy.add(cm, result[1])
			k += 1

		return {"max_accuracy":max(accuracy),
						"mean_accuracy":sum(accuracy)/float(len(accuracy)),
						"min_accuracy":min(accuracy),
						"mean_cm":numpy.around(cm/k, decimals=3)}


	def densifyPattern(self, bitmap):
		"""Return a numpy array of 0s and 1s to represent the input bitmap."""
		densePattern = numpy.zeros(self.n)
		densePattern[bitmap] = 1.0
		return densePattern


	def encodePattern(self, pattern):
		raise NotImplementedError


	def trainModel(self, sample, label):
		raise NotImplementedError


	def testModel(self, sample):
		raise NotImplementedError


################################################################################
# Model checkpoint methods

	def save(self, saveModelDir):  ## TODO (use pickle?)
		"""
		Save the model in the given directory.

		@param saveModelDir 		(string)				Absolute directory path for saving
																						the experiment. If the directory
		        																already exists, it must contain a
		        																valid local checkpoint of a model.
		"""
		modelPklFilePath = self._getModelPklFilePath(saveModelDir)
		
		# Clean up old saved state, if any
		self._cleanSaveModelDirectory(saveModelDir, modelPklFilePath)

		# Create a new directory for saving state
		self._makeSaveModelDirectory(saveModelDir)

		try:
			with open(modelPklFilePath, 'wb') as f:
				pickle.dump(self, f)
		except IOError("Could not open and dump model pickle file."):
			return


	def load(self, loadModelDir):
		"""
		Load saved model.
		@param loadModelDir  (string)			Directory of where the experiment is to 
																		be, or was previously saved.
		@returns 							(Model) 			The loaded model instance
		"""
		modelPklFilePath = self._getModelPklFilePath(loadModelDir)
		try:
			with open(modelPklFilePath, 'rb') as f:
				return pickle.load(f)
		except IOError("Could not open and dump model pickle file."):
			return []


	@staticmethod
	def _getModelPklFilePath(saveModelDir):
		"""
		Return the absolute path ot the model's pickle file.
		@param saveModelDir 	(string)			Directory of where the experiment is to 
	  																	be, or was previously saved.
		@returns 							(string) 			An absolute path.
		"""
		return os.path.abspath(os.path.join(saveModelDir, "model.pkl"))


	@staticmethod
	def _cleanSaveModelDirectory(dirPath, filePath):
		"""
		Cleans directory with saved file, if it exists.

		"""
		if os.path.exists(dirPath):
			assert(os.path.isdir(dirPath),
				"Directory \'{0}\' is not a model checkpoint. Not deleting.".format(
				dirPath))
			assert(os.path.isfile(filePath),
				"File \'{0}\' is not a model checkpoint. Not deleting.".format(
				filePath))

			shutil.rmtree(dirPath)


	@staticmethod
	def _makeSaveModelDirectory(dirPath):
		"""
		Makes directory if it doesn't already exist.

		@param dirPath			 		(str)					Absolute path of directory to create.
		@exception							(Exception)		OSError if directory creation fails.
		"""
		assert os.path.isabs(dirPath)
		try:
			os.makedirs(dirPath)
		except OSError as e:
			if e.errno != os.errno.EEXIST:
				raise
		return


################################################################################
# Model details methods
## TODO: implement these???

	def getInfo(self):
		"""Return info about the model."""


		return {}


	def printReport(self, results): ## TODO
		"""
		"""



		pass


## THESE COMMENTED OUT METHODS ARE SPECIFIC TO HTM, AND SHOULD BE SUBCLASSED ACCORDINGLY

	# def createNetwork():
	#   """Set up a network and return it"""
	#   raise NotImplementedError


	# def feedText(self, text): ##########????????????
	# 	super(ClassificationModel, self).feedText(text)


	# def computeAnomalyScores(self):
	# 	super(ClassificationModel, self).computeAnomalyScores()


	# def resetSequence(self):
	# 	super(ClassificationModel, self).resetSequence()

