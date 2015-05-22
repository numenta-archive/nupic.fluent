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

from fluent.models.model import Model



class ClassificationModel(Model):
	"""
	An NLP model for classification tasks. The subclasses implement methods 
	specific to their tasks/experiments.
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


	def trainModel(self):
		raise NotImplementedError


	def testModel(self):
		raise NotImplementedError


	def runExperiment(self):
		raise NotImplementedError


	def save(self):
		super(ClassificationModel, self).save()


	def load(self):
		super(ClassificationModel, self).load()


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

