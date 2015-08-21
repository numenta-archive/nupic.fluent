# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import cPickle as pkl
import itertools
import numpy
import os
import random

from collections import defaultdict, OrderedDict
from fluent.utils.csv_helper import readCSV, writeFromDict

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

    self.modelDir = os.path.join(
        self.resultsDir, self.experimentName, self.modelName)
    if not os.path.exists(self.modelDir):
      os.makedirs(self.modelDir)

    if self.plots:
      from fluent.utils.plotting import PlotNLP
      self.plotter = PlotNLP()

    self.dataDict = None
    self.labels = None
    self.labelRefs = None
    self.partitions = []
    self.samples = OrderedDict()
    self.patterns = None
    self.results = []
    self.model = None


  def _calculateTrialAccuracies(self):
    """
    @return trialAccuracies     (defaultdict)   Items are defaultdicts, one for
        each size of the training set. Inner defaultdicts keys are
        categories, with numpy array values that contain one accuracy value for
        each trial.
    """
    # To handle multiple trials of the same size:
    # trialSize -> (category -> list of accuracies)
    trialAccuracies = defaultdict(lambda: defaultdict(lambda: numpy.ndarray(0)))
    for i, size in enumerate(self.trainSize):
      accuracies = self.model.calculateClassificationResults(self.results[i])
      for label, acc in accuracies:
        category = self.labelRefs[label]
        accList = trialAccuracies[size][category]
        trialAccuracies[size][category] = numpy.append(accList, acc)

    return trialAccuracies


  def _calculateClassificationAccuracies(self, trialAccuracies):
    """
    @param trialAccuracies            (defaultdict)   Please see the description
        in self._calculateClassificationAccuracies().

    @return classificationAccuracies  (defaultdict)   Keys are classification
        categories, with multiple numpy arrays as values -- one for each size of
        training sets, with one accuracy value for each run of that training set
        size.
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


  def _mapLabelRefs(self):
    """Replace the label strings in self.dataDict with corresponding ints."""
    self.labelRefs = [label for label in set(
        itertools.chain.from_iterable([x[1] for x in self.dataDict.values()]))]

    for uniqueID, data in self.dataDict.iteritems():
      self.dataDict[uniqueID] = (data[0], numpy.array(
          [self.labelRefs.index(label) for label in data[1]]))


  def _preprocess(self, preprocess):
    """Tokenize the samples with or without preprocessing."""
    texter = TextPreprocess()
    if preprocess:
      for uniqueID, data in self.dataDict.iteritems():
        self.samples[uniqueID] = (texter.tokenize(
            data[0], ignoreCommon=100, removeStrings=["[identifier deleted]"],
            correctSpell=True), data[1])
    else:
      for uniqueID, data in self.dataDict.iteritems():
        self.samples[uniqueID] = (texter.tokenize(data[0]), data[1])


  def setupData(self, preprocess=False):
    """
    Get the data from CSV and preprocess if specified.
    One index in labelIdx implies the model will train on a single
    classification per sample.
    @param preprocess   (bool)    Whether or not to preprocess the data when
                                  generating the files
    """
    self.dataDict = readCSV(self.dataPath, self.numClasses)

    if (not isinstance(self.trainSize, list) or not
        all([0 <= size <= len(self.dataDict) for size in self.trainSize])):
      raise ValueError("Invalid size(s) for training set.")

    self._mapLabelRefs()

    self._preprocess(preprocess)

    if self.verbosity > 1:
      for i, s in self.samples.iteritems():
        print i, s


  def initModel(self):
    """Load or instantiate the classification model."""
    if self.load:
      self.loadModel()
    else:
      try:
        module = __import__(self.modelModuleName, {}, {}, self.modelName)
        modelClass = getattr(module, self.modelName)
        self.model = modelClass(
            verbosity=self.verbosity, modelDir=self.modelDir)
      except ImportError:
        raise RuntimeError("Could not import model class \'{0}\'.".
                           format(self.modelName))


  def loadModel(self):
    """Load the serialized model."""
    try:
      with open(os.path.join(self.modelDir, "model.pkl"), "rb") as f:
        model = pkl.load(f)
      print "Model loaded from \'{}\'.".format(self.modelDir)
      return model
    except IOError as e:
      print "Could not load model from \'{}\'.".format(self.modelDir)
      raise e


  def resetModel(self, trial):
    self.model.resetModel()


  def encodeSamples(self):
    self.patterns = self.model.encodeSamples(self.samples)


  def runExperiment(self):
    """Train and test the model for each trial specified by self.trainSize."""
    for i, size in enumerate(self.trainSize):
      self.partitions.append(self.partitionIndices(size, i))

      self.resetModel(i)
      if self.verbosity > 0:
        print "\tTraining for run {0} of {1}.".format(
            i + 1, len(self.trainSize))
      self.training(i)
      if self.verbosity > 0:
        print "\tTesting for this run."
      self.testing(i)


  def training(self, trial):
    """
    Train the model one-by-one on each pattern specified in this trials
    partition of indices. Models' training methods require the sample and label
    to be in a list.
    """
    if self.verbosity > 0:
      print ("\tRunner selects to train on sample(s) {}".format(
          self.partitions[trial][0]))

    for i in self.partitions[trial][0]:
      self.model.trainModel(i)


  def testing(self, trial):
    if self.verbosity > 0:
      print ("\tRunner selects to test on sample(s) {}".format(
          self.partitions[trial][1]))

    results = ([], [])
    for i in self.partitions[trial][1]:
      predicted = self.model.testModel(i)
      results[0].append(predicted)
      results[1].append(self.patterns[i]["labels"])

    self.results.append(results)


  def writeOutClassifications(self):
    """Write the samples, actual, and predicted classes to a CSV."""
    headers = ("Tokenized sample", "Actual", "Predicted")
    for trial, _ in enumerate(self.trainSize):
      resultsDict = defaultdict(list)
      for i, sampleNum in enumerate(self.partitions[trial][1]):
        # Loop through the indices in the test set of this trial.
        sample = self.samples.values()[sampleNum][0]
        pred = sorted([self.labelRefs[j] for j in self.results[trial][0][i]])
        actual = sorted([self.labelRefs[j] for j in self.results[trial][1][i]])
        resultsDict[sampleNum] = (sampleNum, sample, actual, pred)

      resultsPath = os.path.join(self.model.modelDir,
                                 "results_trial" + str(trial) + ".csv")
      writeFromDict(resultsDict, headers, resultsPath)


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
      self.plotter.plotCumulativeAccuracies(
          classificationAccuracies, self.trainSize)

      if self.plots > 1:
        # Plot extra evaluation figures -- confusion matrix.
        self.plotter.plotConfusionMatrix(
            self.setupConfusionMatrices(resultCalcs))


  def partitionIndices(self, split, trial):
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
    dataDict = readCSV(expectationFilePath, self.numClasses)

    accuracies = numpy.zeros((len(self.results)))
    for i, trial in enumerate(self.results):
      for j, predictionList in enumerate(trial[0]):
        predictions = [self.labelRefs[p] for p in predictionList]
        if predictions == []:
          predictions = ["(none)"]
        expected = dataDict.items()[j+self.trainSize[i]][1]

        accuracies[i] += (float(len(set(predictions) & set(expected[1])))
                          / len(expected[1]))

      accuracies[i] = accuracies[i] / len(trial[0])

    return accuracies
