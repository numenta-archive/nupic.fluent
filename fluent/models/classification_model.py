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

  Methods/properties that must be implemented by subclasses:
  	- encodePattern()
  	- trainModel()
  	- testModel()

	"""


	def evaluateTrialResults(self, classifications, n):  ## TODO: add precision, recall, F1 score
		"""
		Calculate statistics for the predicted classifications against the actual.

		@param classifications	(list)						Two lists: (0) predictions and (1)
																							actual classifications.
		@param n 								(int)							Number of classification labels 
																							possible in the dataset.
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


	@staticmethod
	def _printReport(results):  ## TODO: pprint
		"""Prints results as returned by evaluateResults()."""
		print "---------- RESULTS ----------"
		print "max, mean, min accuracies = "
		print "{0:.3f}, {1:.3f}, {2:.3f}".format(
			results["max_accuracy"], results["mean_accuracy"], results["min_accuracy"])
		print "mean confusion matrix =\n", results["mean_cm"]


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
