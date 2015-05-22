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

import csv
import numpy
import os
import random
import time

from collections import Counter
from fluent.bin.utils import getCSVInfo, getFrequentWords, tokenize
# from fluent.encoders.random_encoder import RandomEncoder
from fluent.models.classification_model import ClassificationModel
from nupic.algorithms.KNNClassifier import KNNClassifier



class ClassificationModelRandomSDR(ClassificationModel): ## TODO: unpack args from runner

  def __init__(self,
      dataDir='data',
  		dataPath=None,
      encoder='random',
  		kCV=3,  									# If = 1, no cross-validation
  		resultsPath='',
  		train=True,
  		test=False,
      verbosity=1
  	):

    # Verify input params.
    if not os.path.isfile(dataPath):
      raise ValueError("Invalid data path; either does not exist or there are "
                       "no data files.")
    if (not isinstance(kCV, int)) or (kCV < 1):
      raise ValueError("Invalid value for number of cross-validation folds.")

    self.encoder      = encoder
    self.dataPath     = dataPath
    self.kCV          = kCV
    self.resultsPath  = resultsPath
    self.test         = test
    self.train        = train
    self.verbosity    = verbosity

    # Init kNN classifier:
    # specify 'distanceMethod'='rawOverlap' for overlap; Euclidean is std.
    # pass verbosity=1 for debugging
    self.classifier = KNNClassifier()  

    # SDR dimensions:
    self.n = 16384
    self.w = 328

    # Store list of terms that are cut from the data:
    self.ignore = getFrequentWords()


  def _randomSDR(self, string):  ## better to use a 'random encoder' object?
    """
    Returns a randomly encoded SDR of the input string, w/ same dimensions 
    as the Cio encoder. We seed the random number generator such that a given 
    string will yield the same SDR each time this function is called.
    Saving the internal state of the generator reduces the likelihood of
    repeating values from previous inputs.
    """
    state = random.getstate()
    random.seed(string)
    bitmap = random.sample(xrange(self.n), self.w)
    random.setstate(state)
    return sorted(bitmap)


  def _densifyPattern(self, bitmap):
    """Return a numpy array of 0s and 1s to represent the input bitmap."""
    densePattern = numpy.zeros(self.n)
    densePattern[bitmap] = 1.0
    return densePattern


  def _winningLabel(self, labels):
    """Returns the most frequent item in the input list of labels."""
    data = Counter(labels)
    return data.most_common(1)[0][0]


  def trainModel(self, trainIndices, labels):
    """
    Train the classifier on the input CSV file; expects the following
    formatting:
    - one header row
    - one page
    - headers: 'index', 'question', 'response', 'classifications'

    @param evalIndices    (list)            Indices to specify which lines in
                                            the data file are for training.
    """
    try:
      with open(self.dataPath) as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, line in enumerate(reader):
          if i in trainIndices:
            for sample in tokenize(line[2], ignoreCommon=True):
              # Get a random SDR for each nonempty token, and learn w/ kNN.
              if sample == '': continue
              sampleBitmap = self._randomSDR(sample)
              for label in line[3].split(','):  ## Can kNN handle multiple classes? If so, no loop
                numPatterns = self.classifier.learn(sampleBitmap, labels.index(label), isSparse=self.n)
        if self.verbosity > 0:
          print "Cumulative number of patterns classified = %i." % numPatterns
    except IOError:
      print ("Input file does not exist.")


  def testModel(self, evalIndices):
    """
    Test the classifier on the input CSV file; expects the following
    formatting:
    - one header row
    - one page
    - headers: 'index', 'question', 'response', 'classifications'

    @param evalIndices        (list)        Indices to specify which lines in
                                            the data file are for testing.
    @return classifications   (list)        The 'winner' classifications for the
                                            data samples; for more, see the
                                            KNNClassifier.infer() documentation.
    @return labels            (list)        The true classifications.
    """
    predictLabels = []
    actualLabels = []
    try:
      with open(self.dataPath) as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, line in enumerate(reader):
          tokenLabels = []
          if i in evalIndices:
            for sample in tokenize(line[2], ignoreCommon=True):
              # Get a random SDR for each nonempty token, and infer w/ kNN.
              if sample == '': continue
              sampleBitmap = self._randomSDR(sample)
              (tokenLabel, _, _, _) = self.classifier.infer(
                self._densifyPattern(sampleBitmap))
              tokenLabels.append(tokenLabel)
            # Actual labels are all the same for this line, but predicted
            # label for this line is cumulative across the token labels.
            if tokenLabels == []:
              print "Line skipped b/c kNN returned no classifications."
              continue
            actualLabels.append(line[3].split(','))
            predictLabels.append(self._winningLabel(tokenLabels))
    except IOError:
      print ("Input file does not exist.")

    return predictLabels, actualLabels


  def runExperiment(self):
    """
    Run k-fold cross-validation on a single survey question -- expects one data
    file for the survey question.
    Note: if self.kCV = 1 there will be no cross-validation
    """
    # Get some info about the input data.
    labels, numSamples = getCSVInfo(self.dataPath)

    # Run kCV trials of different data splits for k-fold cross validation.
    split = numSamples/self.kCV  # this will approximate k-folds; partitions the training data based on survey responsees, not the individual tokens
    intermResults = []
    for k in range(self.kCV):
      # Train the model on a subset, and hold the evaluation subset.
      evalIndices = range(k*split, (k+1)*split)
      trainIndices = [i for i in range(numSamples) if not i in evalIndices]

      print "Training for CV fold %i." % k
      self.trainModel(trainIndices, labels)

      print "Evaluating for trial %i." % k
      predicted, actual = self.testModel(evalIndices)

      print "Calculating intermediate results..."
      actual = [labels.index(c[0]) for c in actual]
      intermResults.append(self.evaluateTrialResults(actual, predicted, labels))

    print "Calculating cumulative results for %i trials..." % k
    return self.evaluateResults(intermResults)
