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

from collections import Counter
from fluent.experiments.runner import Runner
from fluent.utils.csv_helper import readCSV
from fluent.utils.network_data_generator import NetworkDataGenerator
from nupic.engine import Network

try:
  import simplejson as json
except:
  import json



class HTMRunner(Runner):
  """
  Class to run the HTM NLP experiments with the specified data, models,
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
               verbosity,
               generateData=True,
               votingMethod="last",
               classificationFile=""):
    """
    @param generateData       (bool)   Whether or not we need to generate data
    @param votingMethod       (string) Either use "last" tokens score or "most"
                                       frequent
    @param classificationFile (string) Path to json file containing the
                                       mappings of string labels to ids
    Look at runner.py for the othr parameters
    """
    self.generateData = generateData
    self.votingMethod = votingMethod

    if classificationFile == "" and not generateData:
      raise ValueError("Must give classificationFile if not generating Data")
    self.classificationFile = classificationFile

    self.dataFiles = []

    super(HTMRunner, self).__init__(dataPath, resultsDir, experimentName, load,
                                    modelName, modelModuleName, numClasses, plots,
                                    orderedSplit, trainSize, verbosity)

    
  def _mapLabelRefs(self):
    """Get the mapping from label strings to the corresponding ints."""
    if os.path.isfile(self.classificationFile) and \
      os.path.splitext(self.classificationFile)[1] == ".json":
      labelToId = json.load(open(self.classificationFile))
      self.labelRefs = zip(*sorted(labelToId.iteritems(), key=lambda x:x[1]))[0]
    else:
      raise ValueError("must have a valid classification json file")


  def setupData(self, preprocess=False, sampleIdx=2):
    """
    Get the data from a directory and preprocess if specified.
    One index in labelIdx implies the model will train on a single
    classification per sample.
    """
    recordStreamDataFile = self.dataPath
    if self.generateData:
      ndg = NetworkDataGenerator()
      ndg.split(self.dataPath, sampleIdx, self.numLabels, preprocess,
        ignoreCommon=100, removeStrings=["[identifier deleted]"],
        correctSpell=True)

      filename, ext = os.path.splitext(self.dataPath)
      self.classificationFile = "{}-classifications.json".format(filename)
      for i in len(self.trainSize):
        if not self.orderedSplit:
          ndg.randomize()
        dataFile = "{}-{}{}".format(filename, i, ext)
        ndg.saveData(dataFile, "{}-classifications.json".format(filename))
        self.dataFiles.append(dataFile)
    else:
      self.dataFiles = [self.dataPath] * len(self.trainSize)

    self._mapLabelRefs()


  def initModel(self, trial=None):
    """Load or instantiate the classification model."""
    if self.load:
      with open(os.path.join(self.modelPath, "model.pkl"), "rb") as f:
        self.model = pkl.load(f)
      networkFile = self.model.network
      self.model.network = Network(networkFile)
      print "Model loaded from \'{0}\'.".format(self.modelPath)
    else:
      try:
        if trial is not None:
          module = __import__(self.modelModuleName, {}, {}, self.modelName)
          modelClass = getattr(module, self.modelName)
          self.model = modelClass(self.dataFiles[trial],
                                  verbosity=self.verbosity)
        else:
          print "Must specify a trial"
      except ImportError:
        raise RuntimeError("Could not find model class \'{0}\' to import.".
                           format(self.modelName))


  def encodeSamples(self):
    """
    Encode the text samples into bitmap patterns. The
    encoded patterns are stored in a dict along with their corresponding class
    labels.
    """
    pass


  def getClassifications(self, split, trial):
    dataFile = self.dataFiles[trial]
    classifications = NetworkDataGenerator.getClassifications(dataFile)
    return [[int(c) for c in classes.split(" ")] for classes in classifications][split:]

  
  def runExperiment(self):
    """Train and test the model for each trial specified by self.trainSize."""
    for i, size in enumerate(self.trainSize):
      self.partitions.append(self.partitionIndices(size, i))

      if self.verbosity > 0:
        print ("\tRunner randomly selects to train on sample(s) {0}, and test "
               "on sample(s) {1}.".
               format(self.partitions[i][0], self.partitions[i][1]))

      self.actualLabels = self.getClassifications(size, i)
      self.initModel(i)
      print "\tTraining for run {0} of {1}.".format(i+1, len(self.trainSize))
      self.training(i)
      print "\tTesting for this run."
      self.testing(i)


  def training(self, trial):
    """
    Train the model one-by-one on each pattern specified in this trials
    partition of indices. Models' training methods require the sample and label
    to be in a list.
    """
    for numTokens in self.partitions[trial][0]:
      for _ in xrange(numTokens):
        self.model.trainModel()


  def _selectWinners(self, predictions):
    if self.votingMethod == "last":
      return predictions[-1]
    elif self.votingMethod == "most":
      counter = Counter()
      for p in predictions:
        counter.update(p)
      return zip(*Counter.most_common(self.numClasses))[0]
    else:
      raise ValueError("voting method must be either \'last\' or \'most\'")


  def testing(self, trial):
    results = ([], [])
    for i, numTokens in enumerate(self.partitions[trial][1]):
      predictions = []
      for _ in xrange(numTokens):
        predicted = self.model.testModel()
        predictions.append(predicted)
      winningPredictions = self._selectWinners(predictions)
      results[0].append(winningPredictions)
      results[1].append(self.actualLabels[i])

    self.results.append(results)


  def save(self):
    # Can't pickle a SWIG object so serialize it using nupic
    networkPath = os.path.join(self.modelPath, "network.nta")
    self.model.network.save(networkPath)
    self.model.network = networkPath
    super(HTMRunner, self).save()


  def partitionIndices(self, split, trial):
    """
    Returns train and test indices.
    """
    dataFile = self.dataFiles[trial]
    resets = NetworkDataGenerator.getResetsIndices(dataFile)
    for i in reversed(xrange(1, len(resets))):
      resets[i] -= resets[i - 1]

    return (resets[:split], resets[split:])
