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

import numpy
import os
import time

try:
  import simplejson as json
except ImportError:
  import json


class ClassificationModel(object):
	"""
	Base class for NLP models of classification tasks. When inheriting from this
  class please take note of which methods MUST be overridden, as documented
  below.

  The Model superclass implements:
    - 
    - ...

  Methods/properties that must be implemented by subclasses:
  	-

	"""

	def evaluateTrialResults(self, actual, predicted, labels):  ## TODO: add precision, recall, F1 score
		"""
		Calculate statistics for the predicted classifications against the actual.

		@param predicted				(list)						Predicted classifications.
		@param actual						(list)						Actual (true) classifications.
		@return									(tuple)						Returns a 2-item tuple w/ the
																							accuracy (float) and confusion
																							matrix (numpy array).
		"""
		if len(predicted) != len(actual):
			raise ValueError("Classification lists must have same length.")
		actual = numpy.array(actual)
		predicted = numpy.array(predicted)
		
		accuracy = (actual == predicted).sum() / float(len(actual))

		cm = numpy.zeros((len(labels), len(labels)))
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


	def encodePattern(self, pattern):
		raise NotImplementedError


	def densifyPattern(self, bitmap):
		"""Return a numpy array of 0s and 1s to represent the input bitmap."""
		densePattern = numpy.zeros(self.n)
		densePattern[bitmap] = 1.0
		return densePattern


	def trainModel(self, trainIndices, labels): ## TODO: for this and all superclass methods that raise notImplError, make sure signature is same as subclass methods
		raise NotImplementedError


	def testModel(self): ## ^^
		raise NotImplementedError


	def runExperiment(self):
		raise NotImplementedError


	def save(self):  ## TODO (use pickle?)
		"""
		Save the model, returning the file path. If saving fails, an empty string is
		returned.
		"""


		return filePath


	def load(self):
		"""Loads a saved model, returning the name."""


		return modelName


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

