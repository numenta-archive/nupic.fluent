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
import os

from collections import Counter, namedtuple
from fluent.experiments.runner import Runner
from fluent.models.classify_htm import ClassificationModelHTM
from fluent.utils.network_data_generator import NetworkDataGenerator
from nupic.engine import Network

try:
  import simplejson as json
except ImportError:
  import json



class HTMRunner(Runner):
  """
  Class to run the HTM NLP experiments with the specified data and evaluation
  metrics.
  """

  def __init__(self,
               dataPath,
               networkConfigPath,
               resultsDir,
               experimentName,
               loadPath,
               modelName,
               numClasses=3,
               plots=0,
               orderedSplit=False,
               trainSizes=None,
               verbosity=0,
               generateData=True,
               votingMethod="last",
               classificationFile=""):
    """
    @param networkConfigPath  (str)    Path to JSON specifying network params.
    @param generateData       (bool)   Whether or not we need to generate data.
    @param votingMethod       (str)    Classify with "last" token's score or
                                       "most" frequent of the sequence.
    @param classificationFile (str)    Path to JSON that maps labels to ids.

    See base class constructor for the other parameters.
    """
    self.networkConfig = self._getNetworkConfig(networkConfigPath)
    self.model = None

    self.generateData = generateData
    self.votingMethod = votingMethod

    if classificationFile == "" and not generateData:
      raise ValueError("Must give classificationFile if not generating data")
    self.classificationFile = classificationFile

    self.dataFiles = []
    self.actualLabels = None

    super(HTMRunner, self).__init__(dataPath, resultsDir, experimentName,
                                    modelName, loadPath, numClasses, plots,
                                    orderedSplit, trainSizes, verbosity)


  @staticmethod
  def _getNetworkConfig(networkConfigPath):
    try:
      with open(networkConfigPath, "rb") as fin:
        return json.load(fin)
    except IOError as e:
      print "Could not find network configuration JSON at \'{}\'.".format(
        networkConfigPath)
      raise e


  def initModel(self):
    """Load or instantiate the classification model."""
    if self.loadPath:
      with open(self.loadPath, "rb") as f:
        self.model = pkl.load(f)
      # TODO: uncomment once we can save TPRegion; do we need this?
      # networkFile = self.model.network
      # self.model.network = Network(networkFile)
      print "Model loaded from \'{0}\'.".format(self.loadPath)
    else:
      self.model = ClassificationModelHTM(self.networkConfig,
                                          self.dataPath,
                                          verbosity=self.verbosity,
                                          numLabels=self.numClasses,
                                          modelDir=self.modelDir)


  def _mapLabelRefs(self):
    """Get the mapping from label strings to the corresponding ints."""
    try:
      with open(self.classificationFile, "r") as f:
        labelToId = json.load(f)
      # Convert the dict of strings -> ids to a list of strings ordered by id
      self.labelRefs = zip(*sorted(labelToId.iteritems(), key=lambda x:x[1]))[0]
    except IOError as e:
      print "Must have a valid classification json file"
      raise e


  def setupData(self, preprocess=False, **kwargs):
    """
    Generate the data in network API format if necessary. self.dataFiles is
    populated with the paths of network data files, one for each trial

    Look at runner.py (setupData) and network_data_generator.py (split) for the
    parameters
    """
    if self.generateData:
      ndg = NetworkDataGenerator()
      ndg.split(self.dataPath, self.numClasses, preprocess, **kwargs)

      filename, ext = os.path.splitext(self.dataPath)
      self.classificationFile = "{}-classifications.json".format(filename)
      for i in xrange(len(self.trainSizes)):
        if not self.orderedSplit:
          ndg.randomizeData()
        dataFile = "{}-{}{}".format(filename, i, ext)
        ndg.saveData(dataFile, self.classificationFile)
        self.dataFiles.append(dataFile)

      if self.verbosity > 0:
        print "{} file(s) generated at {}".format(len(self.dataFiles),
          self.dataFiles)
        print "Classification json is at: {}".format(self.classificationFile)
    else:
      # Does an orderedSplit
      self.dataFiles = [self.dataPath] * len(self.trainSizes)

    if self.numClasses > 0:
      # Setup labels data objects
      self.actualLabels = [self._getClassifications(size, i)
        for i, size in enumerate(self.trainSizes)]
      self._mapLabelRefs()


  def resetModel(self):
    """Load or instantiate the classification model."""
    self.initModel()
    # TODO: change to same as Runner:
    #   self.model.resetModel()
    #   otherwise you're creating a new model instance twice each experiment


  def encodeSamples(self):
    """Passthrough b/c the network encodes the samples."""
    pass


  def _getClassifications(self, split, trial):
    """
    Gets the classifications for testing samples for a particular trial
    @param split      (int)       Size of training set
    @param trial      (int)       trial count
    @return           (list)      List of list of ids of classifications for a
                                  sample
    """
    dataFile = self.dataFiles[trial]
    classifications = NetworkDataGenerator.getClassifications(dataFile)
    return [[int(c) for c in classes.strip().split(" ")]
             for classes in classifications][split:]


  def _training(self, trial):
    """
    Train the network on all the tokens in the training set for a particular
    trial.
    @param trial      (int)       current trial number
    """
    if self.verbosity > 0:
      i = 0
      indices = []
      for numTokens in self.partitions[trial][0]:
        indices.append(i)
        i += numTokens
      print ("\tRunner selects to train on sequences starting at indices {}.".
            format(indices))

    for numTokens in self.partitions[trial][0]:
      self.model.trainModel(iterations=numTokens)


  def _selectWinners(self, predictions):
    """
    Selects the final classifications for the predictions.  Voting
    method=="last" means the predictions of the last sample are used. Voting
    method=="most" means the most frequent sample is used.
    @param predictions    (list)    List of list of possible classifications
    @return               (list)    List of winning classifications
    """
    if self.votingMethod == "last":
      return predictions[-1]
    elif self.votingMethod == "most":
      counter = Counter()
      for p in predictions:
        counter.update(p)
      return zip(*counter.most_common(self.numClasses))[0]
    else:
      raise ValueError("voting method must be either \'last\' or \'most\'")


  def _testing(self, trial):
    """
    Test the network on the test set for a particular trial and store the
    results
    @param trial      (int)       trial count
    """
    if self.verbosity > 0:
      i = sum(self.partitions[trial][0])
      indices = []
      for numTokens in self.partitions[trial][1]:
        indices.append(i)
        i += numTokens
      print "\tRunner selects to test on sample(s) {}".format(indices)

    results = ([], [])
    for i, numTokens in enumerate(self.partitions[trial][1]):
      predictions = []
      for _ in xrange(numTokens):
        predicted = self.model.testModel()
        predictions.append(predicted)
      winningPredictions = self._selectWinners(predictions)

      # TODO: switch to standard (expected, actual) format
      results[0].append(winningPredictions)
      results[1].append(self.actualLabels[trial][i])

    # Prepare data for writeOutClassifications
    trainIdx = range(len(self.partitions[trial][0]))
    testIdx = range(len(self.partitions[trial][0]),
      len(self.partitions[trial][0]) + len(self.partitions[trial][1]))
    self.partitions[trial] = (trainIdx, testIdx)
    self.samples = NetworkDataGenerator.getSamples(self.dataFiles[trial])

    self.results.append(results)


  def saveModel(self):
    """Save the serialized model and network"""
    # Can't pickle a SWIG object so serialize it using nupic
    networkPath = os.path.join(self.modelDir, "network.nta")
    # TODO: uncomment once we can save TPRegion
    #self.model.network.save(networkPath)
    self.model.network = networkPath
    super(HTMRunner, self).saveModel()


  def partitionIndices(self):
    """
    Sets self.partitions for the number of tokens for each sample in the
    training and test sets (when doing an ordered split).
    """
    for trial, split in enumerate(self.trainSizes):
      dataFile = self.dataFiles[trial]
      numTokens = NetworkDataGenerator.getNumberOfTokens(dataFile)
      self.partitions.append((numTokens[:split], numTokens[split:]))


  def partitionLearning(self):
    """
    Find the number of partitions for the input data based on a specific
    networkConfig.

    @return partitions: (list of namedtuples) Region names and index at which the
      region is to begin learning. The final partition is reserved as a test set.
    """
    Partition = namedtuple("Partition", "partName index")

    # Add regions to partition list in order of learning.
    regionConfigs = ("spRegionConfig", "tmRegionConfig", "upRegionConfig",
      "classifierRegionConfig")
    partitions = []

    # TODO
    return


  def writeOutClassifications(self):
    # TODO: implement this method after updating HTM network models and runner
    # per nupic.research #277
    pass
