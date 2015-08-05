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

from classification_network import createNetwork
from fluent.encoders.cio_encoder import CioEncoder
from fluent.models.classification_model import ClassificationModel
from nupic.data.file_record_stream import FileRecordStream



class ClassificationModelHTM(ClassificationModel):
  """
  Class to run the survey response classification task with nupic network

  From the experiment runner, the methods expect to be fed one sample at a time.
  """

  def __init__(self, inputFilePath, verbosity=1, numLabels=3, tmTrainingSize=0):
    self.tmTrainingSize = tmTrainingSize

    super(ClassificationModelHTM, self).__init__(verbosity=verbosity,
      numLabels=numLabels)

    # Initialize Network
    self.encoder = CioEncoder(cacheDir="./experiments/cache")
    self.recordStream = FileRecordStream(streamID=inputFilePath)
    self.network = createNetwork((self.recordStream, "py.LanguageSensor",
      self.encoder, self.numLabels))
    self.network.initialize()
    self.classifierType = "knn" # "cla"

    self.oldClassifications = None
    self.lengthOfCurrentSequence = 0
    self.numTrained = 0


  def encodePattern(self, sample):
    """
    Put each token in its own dictionary with its bitmap

    @param sample     (list)            Tokenized sample, where each item is a
                                        string token.
    @return encodings (list)            The sample text, sparsity, and bitmap
                                        for each token.
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
    """Reset the model by clearing the classifier."""
    self.recordStream.clear()
    self.network = createNetwork((self.recordStream, "py.LanguageSensor",
      self.encoder, self.numLabels))
    self.network.initialize()

    self.numTrained = 0
    self.oldClassifications = None


  def trainModel(self):
    """
    Train the classifier on the input sample and labels.
    """
    sensorRegion = self.network.regions["sensor"]
    spatialPoolerRegion = self.network.regions["SP"]
    temporalMemoryRegion = self.network.regions["TM"]
    spatialPoolerRegion.setParameter("learningMode", True)
    temporalMemoryRegion.setParameter("learningMode", True)

    self.network.run(1)

    # Only train classifier once TM is trained
    if self.numTrained >= self.tmTrainingSize:
      labels = sensorRegion.getOutputData("categoryOut")
      for label in labels:
        if label != -1:
          self._classify(label)
    
    self.numTrained += 1


  def _classify(self, label=None):
    """
    Get the labels from the classifier for the last input
    @param label      (int)     class for learning.  If None, just classify
    @return           (list)    inferred values for each category
    """
    classifierRegion =  self.network.regions["classifier"]
    if self.classifierType == "knn":
      return classifierRegion.getOutputData("categoriesOut")
    elif self.classifierType == "cla":
      # TODO: replace with updated CLA
      temporalMemoryRegion = self.network.regions["TM"]

      activeCells = temporalMemoryRegion.getOutputData("bottomUpOut")
      patternNZ = activeCells.nonzero()[0]
      if label is not None:
        classifierRegion.setParameter("learningMode", True)
        classificationIn = {"bucketIdx": int(label),
                            "actValue": int(label)}
      else:
        classifierRegion.setParameter("learningMode", False)
        classificationIn = {"bucketIdx": None,
                            "actValue": None}

      clResults = classifierRegion.getSelf().customCompute(
        recordNum=self.numTrained, patternNZ=patternNZ,
        classification=classificationIn)

      return clResults[int(classifierRegion.getParameter("steps"))]


  def testModel(self, numLabels=3):
    """
    Test the CLA classifier on the input sample.

    @param numLabels      (int)           Number of predicted classifications.
    @return               (numpy array)   The numLabels most-frequent
                                          classifications for the data sample
    """
    sensorRegion = self.network.regions["sensor"]
    spatialPoolerRegion = self.network.regions["SP"]
    temporalMemoryRegion = self.network.regions["TM"]
    spatialPoolerRegion.setParameter("learningMode", False)
    temporalMemoryRegion.setParameter("learningMode", False)

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
