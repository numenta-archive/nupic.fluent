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

import collections
import cPickle as pkl
import itertools
import numpy
import os
import random

from collections import defaultdict
from fluent.utils.csv_helper import readCSV
from fluent.utils.plotting import PlotNLP

from fluent.utils.text_preprocess import TextPreprocess



class Runner(object):
  """
  Class to run the baseline NLP experiments with the specified data, models,
  text processing, and evaluation metrics.
  """

  def __init__(self,
               dataPath,
               resultsDir,
               experimentName,
               load,
               modelName,
               modelModuleName,
               numClasses,
               plots,
               orderedSplit,
               trainSize,
               verbosity):
    """
    @param dataPath         (str)     Path to raw data file for the experiment.
    @param resultsDir       (str)     Directory where for the results metrics.
    @param experimentName   (str)     Experiment name, used for saving results.
    @param load             (bool)    True if a serialized model is to be
                                      loaded.
    @param modelName        (str)     Name of nupic.fluent Model subclass.
    @param modeModuleName   (str)     Model module -- location of the subclass.
    @param numClasses       (int)     Number of classes (labels) per sample.
    @param plots            (int)     Specifies plotting of evaluation metrics.
    @param orderedSplit     (bool)    Indicates method for splitting train/test
                                      samples; False is random, True is ordered.
    @param trainSize        (str)     Number of samples to use in training.
    @param verbosity        (int)     Greater value prints out more progress.

    """
    self.dataPath = dataPath
    self.resultsDir = resultsDir
    self.experimentName = experimentName
    self.load = load
    self.modelName = modelName
    self.modelModuleName = modelModuleName
    self.numClasses = numClasses
    self.plots = plots
    self.orderedSplit = orderedSplit
    self.trainSize = trainSize
    self.verbosity = verbosity

    self.modelPath = os.path.join(
      self.resultsDir, self.experimentName, self.modelName)
    if not os.path.exists(self.modelPath):
      os.makedirs(self.modelPath)

    if self.plots:
      self.plotter = PlotNLP()

    self.dataDice = None
    self.labels = None
    self.labelRefs = None
    self.partitions = []
    self.samples = None
    self.results = []


  def _mapLabelRefs(self):
    """Replace the label strings in self.dataDict with corresponding ints."""
    self.labelRefs = list(set(
        itertools.chain.from_iterable(self.dataDict.values())))

    for k, v in self.dataDict.iteritems():
      self.dataDict[k] = numpy.array(
          [self.labelRefs.index(label) for label in v])


  def _preprocess(self, preprocess):
    """Tokenize the samples, with or without preprocessing."""
    texter = TextPreprocess()
    if preprocess:
      self.samples = [(texter.tokenize(sample,
                                       ignoreCommon=100,
                                       removeStrings=["[identifier deleted]"],
                                       correctSpell=True),
                      labels) for sample, labels in self.dataDict.iteritems()]
    else:
      self.samples = [(texter.tokenize(sample), labels)
                      for sample, labels in self.dataDict.iteritems()]


  def setupData(self, preprocess=False, sampleIdx=2):
    """
    Get the data from CSV and preprocess if specified.
    One index in labelIdx implies the model will train on a single
    classification per sample.
    """
    self.dataDict = readCSV(self.dataPath, sampleIdx, self.numClasses)

    if not (isinstance(self.trainSize, list) or
        all([0 <= size <= len(self.dataDict) for size in self.trainSize])):
      raise ValueError("Invalid size(s) for training set.")

    self._mapLabelRefs()

    self._preprocess(preprocess)

    if self.verbosity > 1:
      for i, s in enumerate(self.samples): print i, s


  def initModel(self):
    """
    Load or instantiate the classification model.
    TODO: does model need to know if multiclass??
    """
    if self.load:
      with open(os.path.join(self.modelPath, "model.pkl"), "rb") as f:
        self.model = pkl.load(f)
      print "Model loaded from \'{0}\'.".format(self.modelPath)
    else:
      try:
        module = __import__(self.modelModuleName, {}, {}, self.modelName)
        modelClass = getattr(module, self.modelName)
        self.model = modelClass(verbosity=self.verbosity)
      except ImportError:
        raise RuntimeError("Could not find model class \'{0}\' to import.".
                           format(self.modelName))


  def encodeSamples(self):
    """
    Encode the text samples into bitmap patterns, and log to txt file. The
    encoded patterns are stored in a dict along with their corresponding class
    labels.
    """
    self.patterns = [{"pattern": self.model.encodePattern(s[0]),
                     "labels": s[1]}
                     for s in self.samples]
    self.model.logEncodings(self.patterns, self.modelPath)


  def runExperiment(self):
    """Train and test the model for each trial specified by self.trainSize."""
    for i, size in enumerate(self.trainSize):
      self.partitions.append(self.partitionIndices(size))

      if self.verbosity > 0:
        print ("\tRunner randomly selects to train on sample(s) {0}, and test "
               "on sample(s) {1}.".
               format(self.partitions[i][0], self.partitions[i][1]))

      self.model.resetModel()
      print "\tTraining for run {0} of {1}.".format(i+1, len(self.trainSize))
      self.training(i)
      print "\tTesting for this run."
      self.testing(i)


  def training(self, trial):
    """
    Train the model one-by-one on each pattern specified in this trials
    partition of indices.
    """
    for i in self.partitions[trial][0]:
      self.model.trainModel(self.patterns[i]["pattern"],
                            self.patterns[i]["labels"])


  def testing(self, trial):
    results = ([], [])
    for i in self.partitions[trial][1]:
      predicted = self.model.testModel(self.patterns[i]["pattern"])
      results[0].append(predicted)
      results[1].append(self.patterns[i]["labels"])

    self.results.append(results)


  def calculateResults(self):
    """
    Calculate evaluation metrics from the result classifications.

    TODO: pass intended CM results to plotter.plotConfusionMatrix()
    """
    resultCalcs = [self.model.evaluateResults(self.results[i],
                                              self.labelRefs,
                                              self.partitions[i][1])
                   for i in xrange(len(self.trainSize))]

    self.model.printFinalReport(self.trainSize, [r[0] for r in resultCalcs])

    if self.plots:
      trialAccuracies = self._calculateTrialAccuracies()
      classificationAccuracies = self._calculateClassificationAccuracies(
          trialAccuracies)

      self.plotter.plotCategoryAccuracies(trialAccuracies, self.trainSize)
      self.plotter.plotCumulativeAccuracies(classificationAccuracies,
          self.trainSize)

      if self.plots > 1:
        # Plot extra evaluation figures -- confusion matrix.
        self.plotter.plotConfusionMatrix(
            model.calculateConfusionMatrix(trialAccuracies))


  def _calculateTrialAccuracies(self):
    """
    """
    # To handle multiple trials of the same size:
    # trialSize -> (category -> list of accuracies)
    trialAccuracies = defaultdict(lambda: defaultdict(lambda:
        numpy.ndarray(0)))
    for i, size in enumerate(self.trainSize):
      accuracies = self.model.calculateClassificationResults(self.results[i])
      for label, acc in accuracies:
        category = self.labelRefs[label]
        acc_list = trialAccuracies[size][category]
        trialAccuracies[size][category] = numpy.append(acc_list, acc)

    return trialAccuracies


  def _calculateClassificationAccuracies(self, trialAccuracies):
    """
    """
    # Need the accuracies to be ordered for the plot
    trials = sorted(set(self.trainSize))
    # category -> list of list of accuracies
    classificationAccuracies = defaultdict(list)
    for trial in trials:
      accuracies = trialAccuracies[trial]
      for label, acc in accuracies.iteritems():
        classificationAccuracies[label].append(acc)

    return classificationAccuracies


  def save(self):
    """Save the serialized model."""
    print "Saving model to \'{0}\' directory.".format(self.modelPath)
    with open(os.path.join(self.modelPath, "model.pkl"), "wb") as f:
      pkl.dump(self.model, f)


  def partitionIndices(self, split):
    """
    Returns train and test indices.

    TODO: use StandardSplit in data_split.py
    """
    length = len(self.samples)
    if self.orderedSplit:
      trainIdx = range(split)
      testIdx = range(split, length)
    else:
      # Randomly sampled, not repeated
      trainIdx = random.sample(xrange(length), split)
      testIdx = [i for i in xrange(length) if i not in trainIdx]

    return (trainIdx, testIdx)


  def validateExperiment(self, expectationFilePath):
    """Returns accuracy of predicted labels against expected labels."""
    dataDict = readCSV(expectationFilePath, 2, self.numClasses)

    accuracies = numpy.zeros((len(self.results)))
    for i, trial in enumerate(self.results):
      for j, predictionList in enumerate(trial[0]):
        predictions = [self.labelRefs[p] for p in predictionList if p]
        expected = dataDict.items()[j+self.trainSize[i]][1]
        accuracies[i] += (float(len(set(predictions) & set(expected)))
                          / len(expected))
      accuracies[i] = accuracies[i] / len(trial[0])

    return accuracies
