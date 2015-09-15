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

import numpy
import os

from classification_network import configureNetwork
from fluent.encoders.cio_encoder import CioEncoder
from fluent.models.classification_model import ClassificationModel
from fluent.utils.network_data_generator import NetworkDataGenerator
from nupic.data.file_record_stream import FileRecordStream



class ClassificationModelHTM(ClassificationModel):
  """
  Class to run the survey response classification task with nupic network
  """

  def __init__(self,
               networkConfig,
               inputFilePath,
               verbosity=1,
               numLabels=3,
               modelDir="ClassificationModelHTM",
               prepData=True):
    """
    @param networkConfig      (str)     Path to JSON of network configuration,
                                        with region parameters.
    @param inputFilePath      (str)     Path to data file.

    See ClassificationModel for remaining parameters.
    """

    super(ClassificationModelHTM, self).__init__(
      verbosity=verbosity, numLabels=numLabels, modelDir=modelDir)

    self.networkConfig = networkConfig

    if prepData:
      self.networkDataPath = self.prepData(inputFilePath)
    else:
      self.networkDataPath = inputFilePath

    self.network = self.initModel()
    self.learningRegions = self._getLearningRegions()


  def prepData(self, dataPath, **kwargs): # work w/ Runner setupData()
    """
    Generate the data in network API format.

    @param dataPath     (str)     Path to input data file; format as expected by
                                  NetworkDataGenerator.
    """
    ndg = NetworkDataGenerator()
    return ndg.setupData(dataPath, self.numLabels, ordered, **kwargs)


  def initModel(self):
    """
    Initialize the network; self.networdDataPath must already be set.
    """
    recordStream = FileRecordStream(streamID=self.networkDataPath)
    encoder = CioEncoder(cacheDir="./experiments/cache")

    return configureNetwork(recordStream, self.networkConfig, encoder)


  def _getLearningRegions(self):
    """Return tuple of the network's region objects that learn."""
    learningRegions = []
    for region in self.network.regions.values():
      try:
        _ = region.getParameter("learningMode")
        learningRegions.append(region)
      except:
        continue

    return learningRegions


  def encodeSample(self, sample):  # TODO: does this ever get called??
    """
    Put each token in its own dictionary with its bitmap
    @param sample     (list)            Tokenized sample, where each item is a
                                        string token.
    @return           (list)            The sample text, sparsity, and bitmap
                                        for each token. Since the network will
                                        do the actual encoding, the bitmap and
                                        sparsity will be None
    Example return list:
      [{
        "text": "Example text",
        "sparsity": 0.0,
        "bitmap": None
      }]
    """
    return [{"text": t,
             "sparsity": None,
             "bitmap": None} for t in sample]


  def resetModel(self):
    """
    Reset the model by creating a new network since the network API does not
    support resets.
    """
    # TODO: test this works as expected
    self.network = self.initModel()


  def saveModel(self):
    import pdb; pdb.set_trace()
    try:
      if not os.path.exists(self.modelDir):
        os.makedirs(self.modelDir)
      networkPath = os.path.join(self.modelDir, "network.nta")
      with open(networkPath, "wb") as f:
        pkl.dump(self, f)
      if self.verbosity > 0:
        print "Model saved to \'{}\'.".format(networkPath)
    except IOError as e:
      print "Could not save model to \'{}\'.".format(networkPath)
      raise e


  def trainModel(self, iterations=1):
    """
    Run the network with all regions learning.
    Note self.sampleReference doesn't get populated b/c in a network model
    there's a 1-to-1 mapping of training samples.
    """
    for region in self.learningRegions:
      region.setParameter("learningMode", True)

    classifierRegion = self.network.regions[
      self.networkConfig["classifierRegionConfig"].get("regionName")]
    self.network.run(iterations)


  def testModel(self, numLabels=3):
    """
    Test the classifier region on the input sample. Call this method for each
    word of a sequence.

    @param numLabels  (int)           Number of classification predictions.
    @return           (numpy array)   numLabels most-frequent classifications
                                      for the data samples; int or empty.
    """
    sensorRegion = self.network.regions[
      self.networkConfig["sensorRegionConfig"].get("regionName")]
    classifierRegion = self.network.regions[
      self.networkConfig["classifierRegionConfig"].get("regionName")]

    for region in self.learningRegions:
      region.setParameter("learningMode", False)
    classifierRegion.setParameter("inferenceMode", True)

    self.network.run(1)

    return self._getClassifierInference(classifierRegion)


  def _getClassifierInference(self, classifierRegion):
    """Return output categories from the classifier region."""
    relevantCats = classifierRegion.getParameter("categoryCount")

    if classifierRegion.type == "py.KNNClassifierRegion":
      # max number of inferences = k
      inferenceValues = classifierRegion.getOutputData("categoriesOut")[:relevantCats]
      return self.getWinningLabels(inferenceValues, numLabels=3)


    elif classifierRegion.type == "py.CLAClassifierRegion":
      # TODO: test this
      return classifierRegion.getOutputData("categoriesOut")[:relevantCats]
