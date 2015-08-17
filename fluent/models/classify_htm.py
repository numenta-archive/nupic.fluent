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

from classification.classification_network import createNetwork
from fluent.encoders.cio_encoder import CioEncoder
from fluent.models.classification_model import ClassificationModel
from nupic.data.file_record_stream import FileRecordStream



class ClassificationModelHTM(ClassificationModel):
  """
  Class to run the survey response classification task with nupic network
  """

  def __init__(self, inputFilePath, verbosity=1, numLabels=3, spTrainingSize=0,
               tmTrainingSize=0, clsTrainingSize=0, classifierType="KNN"):
    """
    @param inputFilePath      (str)       Path to data formatted for network
                                          API
    @param spTrainingSize     (int)       Number of samples the network has to
                                          be trained on before training the
                                          spatial pooler
    @param tmTrainingSize     (int)       Number of samples the network has to
                                          be trained on before training the
                                          temporal memory
    @param clsTrainingSize    (int)       Number of samples the network has to
                                          be trained on before training the
                                          classifier
    @param classifierType     (str)       Either "KNN" or "CLA"
    See ClassificationModel for remaining parameters
    """
    self.spTrainingSize = spTrainingSize
    self.tmTrainingSize = tmTrainingSize
    self.clsTrainingSize = clsTrainingSize

    super(ClassificationModelHTM, self).__init__(verbosity=verbosity,
      numLabels=numLabels)

    # Initialize Network
    self.classifierType = classifierType
    self.recordStream = FileRecordStream(streamID=inputFilePath)
    self.encoder = CioEncoder(cacheDir="./experiments/cache")
    self._initModel()


  def _initModel(self):
    """Initialize the network and related variables"""
    if self.classifierType == "CLA":
      classifier_params = {
        "steps": "1",
        "implementation": "py",
        "clVerbosity": self.verbosity
      }
    elif self.classifierType == "KNN":
      classifier_params = {
        "k": self.numLabels,
        "distThreshold": 0,
        "maxCategoryCount": self.numLabels
      }
    else:
      raise ValueError("Classifier type {} is not supported.".format(
        self.classifierType))

    self.network = createNetwork(
      self.recordStream, "py.LanguageSensor", self.encoder, self.numLabels,
      "py.{}ClassifierRegion".format(self.classifierType), classifier_params)

    self.network.initialize()

    spatialPoolerRegion = self.network.regions["SP"]
    temporalMemoryRegion = self.network.regions["TM"]
    classifierRegion = self.network.regions["classifier"]

    spatialPoolerRegion.setParameter("learningMode", False)
    temporalMemoryRegion.setParameter("learningMode", False)
    classifierRegion.setParameter("learningMode", False)

    self.numTrained = 0
    self.oldClassifications = None
    self.lengthOfCurrentSequence = 0


  def encodePattern(self, sample):
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
    self.recordStream.clear()
    self._initModel()


  def trainModel(self):
    """
    Train the network on the input to FileRecordStream.  Train the spatial
    pooler if the network has been trained on enough (self.spTrainingSize)
    samples. Train the temporal memory if the network has been trained on
    enough (self.tmTrainingSize) samples. Train the classifier if the network
    has been trained on enough (self.clsTrainingSize) samples.
    """
    sensorRegion = self.network.regions["sensor"]
    spatialPoolerRegion = self.network.regions["SP"]
    temporalMemoryRegion = self.network.regions["TM"]
    classifierRegion = self.network.regions["classifier"]

    if self.numTrained >= self.spTrainingSize:
      spatialPoolerRegion.setParameter("learningMode", True)
    if self.numTrained >= self.tmTrainingSize:
      temporalMemoryRegion.setParameter("learningMode", True)
    if self.numTrained >= self.clsTrainingSize:
      classifierRegion.setParameter("learningMode", True)

    self.network.run(1)

    # TODO: delete after Marion's PR is merged
    # https://github.com/numenta/nupic/pull/2415
    if self.numTrained >= self.clsTrainingSize:
      labels = sensorRegion.getOutputData("categoryOut")
      for label in labels:
        if label != -1:
          self._classify(label)

    self.numTrained += 1


  # TODO: delete after Marion's PR is merged
  def _classify(self, label=None):
    """
    Work around to get the labels from the classifier for the last input
    @param label      (int)     class for learning.  If None, just classify
    @return           (list)    inferred values for each category
    """
    classifierRegion = self.network.regions["classifier"]

    if self.classifierType == "CLA":
      if label is not None:
        classifierRegion.setParameter("learningMode", True)
        classificationIn = {"bucketIdx": int(label),
                            "actValue": int(label)}
      else:
        classifierRegion.setParameter("learningMode", False)
        classificationIn = {"bucketIdx": None,
                            "actValue": None}

      temporalMemoryRegion = self.network.regions["TM"]

      activeCells = temporalMemoryRegion.getOutputData("bottomUpOut")
      patternNZ = activeCells.nonzero()[0]
      clResults = classifierRegion.getSelf().customCompute(
        recordNum=self.numTrained, patternNZ=patternNZ,
        classification=classificationIn)

      return clResults[int(classifierRegion.getParameter("steps"))]

    return classifierRegion.getOutputData("categoriesOut")


  def testModel(self, numLabels=3):
    """
    Test the KNN/CLA classifier on the input sample.
    @param numLabels      (int)           Number of predicted classifications.
    @return               (numpy array)   The numLabels most-frequent
                                          classifications for the data sample
    """
    sensorRegion = self.network.regions["sensor"]
    spatialPoolerRegion = self.network.regions["SP"]
    temporalMemoryRegion = self.network.regions["TM"]
    classifierRegion = self.network.regions["classifier"]

    spatialPoolerRegion.setParameter("learningMode", False)
    temporalMemoryRegion.setParameter("learningMode", False)
    classifierRegion.setParameter("learningMode", False)

    self.network.run(1)

    inferredValue = self._classify()
    reset = sensorRegion.getOutputData("resetOut")[0]

    # TODO: Hard coded for equal weighting. Use lengthOfCurrentSequence later
    i = 1
    if reset or self.oldClassifications is None:
      self.oldClassifications = numpy.array(inferredValue)
      self.lengthOfCurrentSequence = 1
    else:
      self.lengthOfCurrentSequence += 1
      self.oldClassifications += (numpy.array(inferredValue) * i)

    orderedInferredValues = sorted(enumerate(self.oldClassifications),
      key=lambda x: x[1], reverse=True)

    labels = zip(*orderedInferredValues)[0]
    return numpy.array(labels[:numLabels])
