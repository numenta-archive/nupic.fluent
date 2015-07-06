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

import copy
import numpy
import os
import pandas
import random

from collections import Counter
from fluent.utils.plotting import PlotNLP

try:
  import simplejson as json
except ImportError:
  import json



class ClassificationModel(object):  ## TODO: update docstring
  """
  Base class for NLP models of classification tasks. When inheriting from this
  class please take note of which methods MUST be overridden, as documented
  below.

  The Model superclass implements:
    - classifyRandomly() calculates accuracy of a random classifier
    - encodeRandomly() creates a random SDR encoding
    - evaluateTrialResults() calculates result stats
    - evaluateResults() calculates result stats for a list of trial results
    - printTrialReport() prints classifications of an evaluation trial
    - printFinalReport() prints evaluation metrics and confusion matrix
    - densifyPattern() returns a binary SDR vector for a given bitmap

  Methods/properties that must be implemented by subclasses:
    - encodePattern(); note the specified format in the docstring below.
    - trainModel()
    - testModel()
  """

  def __init__(self, n=16384, w=328, verbosity=1, plot=True, multiclass=False):
    """The SDR dimensions are standard for Cortical.io fingerprints."""
    self.n = n
    self.w = w
    self.multiclass = multiclass
    self.verbosity = verbosity
    self.plot = plot


  def encodeRandomly(self, sample):
    """Return a random bitmap representation of the sample."""
    random.seed(sample)
    return numpy.sort(random.sample(xrange(self.n), self.w))


  def logEncodings(self, patterns, path):
    """Log the encoding dictionaries to a txt file."""
    if not os.path.isdir(path):
      raise ValueError("Invalid path to write file.")

    # Cast numpy arrays to list objects for serialization.
    jsonPatterns = copy.deepcopy(patterns)
    for jp in jsonPatterns:
      jp[0]["bitmap"] = jp[0].get("bitmap", None).tolist()

    with open(os.path.join(path, "encoding_log.txt"), "w") as f:
      f.write(json.dumps(jsonPatterns, indent=1))


  def classifyRandomly(self, labels):
    """Return accuracy of random classifications for the labels."""
    randomLabels = numpy.random.randint(0, labels.max(), labels.shape)
    return (randomLabels == labels).sum() / float(labels.shape[0])


  def _densifyPattern(self, bitmap):
    """Return a numpy array of 0s and 1s to represent the input bitmap."""
    densePattern = numpy.zeros(self.n)
    for i in bitmap:
      densePattern[i] = 1.0
    return densePattern

  def _getTopLabels(self, inferenceResult, numLabels=1):
    """Returns the `n` most frequent labels in k nearest neighbors.

    @param inferenceResult (numpy.array) Stores frequency of each category

    @param numLabels (int) Return this number of most frequent labels
          within top k

    @return (numpy.array)  Return list of top `numLabels` labels

  """
    return inferenceResult.argsort()[::-1][:numLabels]


  @staticmethod
  def winningLabels(labels, n=3):
    """
    Returns the most frequent item in the input list of labels. If there are
    ties for the most frequent item, the x most frequent are returned,
    where x<=n.
    """
    labelCount = Counter(labels).most_common()
    maxCount = 0
    for c in labelCount:  ## TODO: better way to do this? -- most_common() should return a reverse sorted list so this is unnecessary
      if c[1] > maxCount:
        maxCount = c[1]
    winners = [c[0] for c in labelCount if c[1]==maxCount]

    return winners if len(winners) <= n else winners[:n]


  def calculateClassificationResults(self, classifications):  ## TODO: plot
    """Calculate the classification accuracy for each category.
    """
    ## TODO


  def evaluateResults(self, classifications, references, idx): ## TODO: evaluation metrics for multiple classifcations
    """
    Calculate statistics for the predicted classifications against the actual.

    @param classifications  (list)            Two lists: (0) predictions and (1)
                                              actual classifications. Items in
                                              the predictions list are lists of
                                              ints or None, and items in actual
                                              classifications list are ints.
    @param references       (list)            Classification label strings.
    @param idx              (list)            Indices from unknown `partitions` variable
    @return                 (tuple)           Returns a 2-item tuple w/ the
                                              accuracy (float) and confusion
                                              matrix (numpy array).
    """
    if self.verbosity > 0:
      self.printTrialReport(classifications, references, idx)

    accuracy = self.calculateAccuracy(classifications)
    cm = self.calculateConfusionMatrix(classifications, references)

    return (accuracy, cm)


  def evaluateCumulativeResults(self, intermResults):
    """
    Cumulative statistics for the outputs of evaluateTrialResults().

    @param intermResults      (list)          List of returned results from
                                              evaluateTrialResults().
    @return                   (dict)          Returns a dictionary with entries
                                              for max, mean, and min accuracies,
                                              and the mean confusion matrix.
    """
    accuracy = []
    cm = numpy.zeros((intermResults[0][1].shape))

    # Find mean, max, and min values for the metrics.
    for result in intermResults:
      accuracy.append(result[0])
      cm = numpy.add(cm, result[1])

    results = {"max_accuracy":max(accuracy),
               "mean_accuracy":sum(accuracy)/float(len(accuracy)),
               "min_accuracy":min(accuracy),
               "total_cm":cm}

    if self.verbosity > 0:
      self.printCumulativeReport(results)

    if self.plot:
      self.plotConfusionMatrix(cm)

    return results


  @staticmethod
  def calculateAccuracy(classifications):   ## TODO: just get numpy arrays passed in?
    """
    Returns classification accuracy -- i.e. correct labels out of total labels.
    """
    if len(classifications[0]) != len(classifications[1]):
      raise ValueError("Classification lists must have same length.")

    actual = numpy.array(classifications[1])
    predicted = numpy.array([c for c in classifications[0]])  ## TODO: multiclass; this forces evaluation metrics to consider only the first predicted classification

    accuracy = 0.0
    for idx in range(len(actual)):
      commonElems = numpy.intersect1d(actual[idx], predicted[idx])
      accuracy += len(commonElems)/float(len(actual[idx]))

    return accuracy/len(actual)

  @staticmethod
  def calculateConfusionMatrix(classifications, references):  ## TODO: just get numpy arrays passed in?
    """Returns confusion matrix as a pandas dataframe."""
    if len(classifications[0]) != len(classifications[1]):
      raise ValueError("Classification lists must have same length.")

    actual = numpy.array(classifications[1])
    predicted = numpy.array([c for c in classifications[0]])  ## TODO: multiclass; this forces evaluation metrics to consider only the first predicted classification

    total = len(references)
    cm = numpy.zeros((total, total+1))
    for i, p in enumerate(predicted):
      if p is not None:
        for aLabel in actual[i]:
          for pLabel in p:
            cm[aLabel][pLabel] += 1
      else:
        # No predicted label, so increment the "(none)" column.
        for aLabel in actual[i]:
          cm[aLabel][total] += 1
    cm = numpy.vstack((cm, numpy.sum(cm, axis=0)))
    cm = numpy.hstack((cm, numpy.sum(cm, axis=1).reshape(total+1,1)))

    cm = pandas.DataFrame(
      data=cm,
      columns=references+["(none)"]+["Actual Totals"],
      index=references+["Prediction Totals"])

    return cm


  @staticmethod
  def printTrialReport(labels, refs, idx):  ## TODO: pprint
    """Print columns for sample #, actual label, and predicted label."""
    template = "{0:<10}|{1:<30}|{2:<30}"
    print "Evaluation results for the trial:"
    print template.format("#", "Actual", "Predicted")
    for i in xrange(len(labels[0])):
      if labels[0][i][0] == None:
        print template.format(idx[i], [refs[label] for label in labels[1][i]], "(none)")
      else:
        print template.format(
          idx[i], [refs[label] for label in labels[1][i]], [refs[label] for label in labels[0][i]])


  @staticmethod
  def printCumulativeReport(results):  ## TODO: pprint
    """
    Prints results as returned by evaluateFinalResults() after several trials.
    """
    print "---------- RESULTS ----------"
    print "max, mean, min accuracies = "
    print "{0:.3f}, {1:.3f}, {2:.3f}".format(
    results["max_accuracy"], results["mean_accuracy"], results["min_accuracy"])
    print "total confusion matrix =\n", results["total_cm"]


  @staticmethod
  def printFinalReport(trainSize, accuracies):  ## TODO: pprint
    """Prints result accuracies."""
    template = "{0:<20}|{1:<10}"
    print "Evaluation results for this experiment:"
    print template.format("Size of training set", "Accuracy")
    for i, a in enumerate(accuracies):
      print template.format(trainSize[i], a)


  @staticmethod
  def plotConfusionMatrix(cm):
    """Output plotly confusion matrix."""
    PlotNLP().confusionMatrix(cm)


  def encodePattern(self, pattern):
    """
    The subclass implementations must return the encoding in the following
    format:
      {
        ["text"]:sample,
        ["sparsity"]:sparsity,
        ["bitmap"]:bitmapSDR
      }
    Note: sample is a string, sparsity is float, and bitmapSDR is a numpy array.
    """
    raise NotImplementedError


  def resetModel(self):
    raise NotImplementedError


  def trainModel(self, sample, label):
    raise NotImplementedError


  def testModel(self, sample, numLabels=1):
    raise NotImplementedError
