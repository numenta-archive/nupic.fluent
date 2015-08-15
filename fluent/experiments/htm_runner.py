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

import cPickle as pkl
import os

from collections import Counter
from fluent.experiments.runner import Runner
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
               classificationFile="",
               classifierType="KNN"):
    """
    @param generateData       (bool)   Whether or not we need to generate data
    @param votingMethod       (str)    Either use "last" tokens score or "most"
                                       frequent
    @param classificationFile (str)    Path to json file containing the
                                       mappings of string labels to ids
    @param classifierType     (str)    Either "KNN" or "CLA"
    Look at runner.py for the other parameters
    """
    self.generateData = generateData
    self.votingMethod = votingMethod
    self.classifierType = classifierType

    if classificationFile == "" and not generateData:
      raise ValueError("Must give classificationFile if not generating data")
    self.classificationFile = classificationFile

    self.dataFiles = []
    self.actualLabels = None

    super(HTMRunner, self).__init__(dataPath, resultsDir, experimentName, load,
                                    modelName, modelModuleName, numClasses,
                                    plots, orderedSplit, trainSize, verbosity)


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


  def setupData(self, preprocess=False, sampleIdx=2, **kwargs):
    """
    Generate the data in network API format if necessary. self.dataFiles is
    populated with the paths of network data files, one for each trial

    Look at runner.py (setupData) and network_data_generator.py (split) for the
    parameters
    """
    if self.generateData:
      ndg = NetworkDataGenerator()
      ndg.split(self.dataPath, sampleIdx, self.numClasses, preprocess,
        **kwargs)

      filename, ext = os.path.splitext(self.dataPath)
      self.classificationFile = "{}-classifications.json".format(filename)
      for i in xrange(len(self.trainSize)):
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
      self.dataFiles = [self.dataPath] * len(self.trainSize)

    self.actualLabels = [self._getClassifications(size, i)
      for i, size in enumerate(self.trainSize)]

    self._mapLabelRefs()


  def resetModel(self, trial):
    """Load or instantiate the classification model."""
    if self.load:
      with open(os.path.join(self.modelPath, "model.pkl"), "rb") as f:
        self.model = pkl.load(f)
      networkFile = self.model.network
      # TODO: uncomment once we can save TPRegion
      #self.model.network = Network(networkFile)
      print "Model loaded from \'{0}\'.".format(self.modelPath)
    else:
      try:
        module = __import__(self.modelModuleName, {}, {}, self.modelName)
        modelClass = getattr(module, self.modelName)
        tmTrainingSize = self.trainSize[trial] / 3.0
        clsTrainingSize = 2 * self.trainSize[trial] / 3.0
        # import pdb; pdb.set_trace()
        self.model = modelClass(self.dataFiles[trial],
                                verbosity=self.verbosity,
                                tmTrainingSize=tmTrainingSize,
                                clsTrainingSize=clsTrainingSize,
                                classifierType=self.classifierType)
      except ImportError:
        raise RuntimeError("Could not import model class \'{0}\'.".
                           format(self.modelName))


  def encodeSamples(self):
    """This method does nothing since the network encodes the samples"""
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


  def training(self, trial):
    """
    Train the network on all the tokens in the training set for a particular
    trial
    @param trial      (int)       trial count
    """
    if self.verbosity > 0:
      i = 0
      indices = []
      for numTokens in self.partitions[trial][0]:
        indices.append(i)
        i += numTokens
      print "\tRunner selects to train on sample(s) {}".format(indices)

    for numTokens in self.partitions[trial][0]:
      for _ in xrange(numTokens):
        self.model.trainModel()


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


  def testing(self, trial):
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
      results[0].append(winningPredictions)
      results[1].append(self.actualLabels[trial][i])

    # Prepare data for writeOutClassifications
    trainIdx = range(len(self.partitions[trial][0]))
    testIdx = range(len(self.partitions[trial][0]),
      len(self.partitions[trial][0]) + len(self.partitions[trial][1]))
    self.partitions[trial] = (trainIdx, testIdx)
    self.samples = NetworkDataGenerator.getSamples(self.dataFiles[trial])

    self.results.append(results)


  def save(self):
    """Save the serialized model and network"""
    # Can't pickle a SWIG object so serialize it using nupic
    networkPath = os.path.join(self.modelPath, "network.nta")
    # TODO: uncomment once we can save TPRegion
    #self.model.network.save(networkPath)
    self.model.network = networkPath
    super(HTMRunner, self).save()


  def partitionIndices(self, split, trial):
    """
    Returns the number of tokens for each sample in the training and test set
    when doing an ordered split
    """
    dataFile = self.dataFiles[trial]
    numTokens = NetworkDataGenerator.getNumberOfTokens(dataFile)
    return (numTokens[:split], numTokens[split:])
