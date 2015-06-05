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

from collections import Counter



class ClassificationModel(object):
	"""
	Base class for NLP models of classification tasks. When inheriting from this
	class please take note of which methods MUST be overridden, as documented
	below.

	The Model superclass implements:
		- evaluateTrialResults() calcualtes result stats
		- evaluateResults() calculates result stats for a list of trial results
		- printTrialReport() prints classifications of an evaluation trial
		- printFinalReport() prints evaluation metrics and confusion matrix
		- densifyPattern() returns a binary SDR vector for a given bitmap

	Methods/properties that must be implemented by subclasses:
		- encodePattern()
		- trainModel()
		- testModel()
	"""

	def __init__(self, verbosity=1):
		self.verbosity = verbosity


	def evaluateTrialResults(self, classifications, references, idx):  ## TODO: add precision, recall, F1 score
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

		if self.verbosity > 0:
			self._printTrialReport(classifications, references, idx)

		accuracy = (actual == predicted).sum() / float(len(actual))

		cm = numpy.zeros((len(references), len(references)))
		for a, p in zip(actual, predicted):
			cm[a][p] += 1

		return (accuracy, cm)


	def evaluateFinalResults(self, intermResults):
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

		results = {"max_accuracy":max(accuracy),
					 		 "mean_accuracy":sum(accuracy)/float(len(accuracy)),
							 "min_accuracy":min(accuracy),
							 "mean_cm":numpy.around(cm/k, decimals=3)}

		if self.verbosity > 0:
			self._printFinalReport(results)

		return results


	@staticmethod
	def _printTrialReport(labels, refs, idx):
		"""Print columns for sample #, actual label, and predicted label."""
		template = "{0:5}|{1:20}|{2:20}"
		print "Evaluation results for this fold:"
		print template.format("#", "Actual", "Predicted")
		for i in xrange(len(labels[0])):
			print template.format(idx[i], refs[labels[1][i]], refs[labels[0][i]])


	@staticmethod
	def _printFinalReport(results):  ## TODO: pprint
		"""Prints results as returned by evaluateResults()."""
		print "---------- RESULTS ----------"
		print "max, mean, min accuracies = "
		print "{0:.3f}, {1:.3f}, {2:.3f}".format(
		results["max_accuracy"], results["mean_accuracy"], results["min_accuracy"])
		print "mean confusion matrix =\n", results["mean_cm"]


	def _densifyPattern(self, bitmap):
		"""Return a numpy array of 0s and 1s to represent the input bitmap."""
		densePattern = numpy.zeros(self.n)
		densePattern[bitmap] = 1.0
		return densePattern


	def _winningLabel(self, labels):
		"""Returns the most frequent item in the input list of labels."""
		data = Counter(labels)
		return data.most_common(1)[0][0]


	def encodePattern(self, pattern):
		raise NotImplementedError


	def resetModel(self):
		raise NotImplementedError


	def trainModel(self, sample, label):
		raise NotImplementedError


	def testModel(self, sample):
		raise NotImplementedError
