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
from fluent.models.classification_model import ClassificationModel
from nupic.data.file_record_stream import FileRecordStream



class ClassificationModelHTM(ClassificationModel):
  """
  Class to run the survey response classification task with nupic network

  From the experiment runner, the methods expect to be fed one sample at a time.
  """

  def __init__(self, inputFilePath, verbosity=1, numLabels=3):
    super(ClassificationModelFingerprint, self).__init__(verbosity, numLabels)

    # Initialize Network
    self.recordStream = FileRecordStream(streamID=inputFilePath)
    self.network = createNetwork((self.recordStream, "py.LanguageSensor",
      self.encoder))
    self.network.initialize()

    self.numTrained = 0
    self.tmTrainingSize = 0
    self.labels = set()
    self.oldClassifications = None
    self.lengthOfCurrentSequence = 0


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
    self.network = createNetwork((self.recordStream, "py.RecordSensor", self.encoder))
    self.network.initialize()

    self.numTrained = 0
    self.labels.clear()
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
    if self.sequenceId > self.tmTrainingSize:
      labels = sensorRegion.getOutputData("categoryOut")[0]
      for label in labels:
        self._classify(label)
    
    self.numTrained += 1


  # TODO: replace with updated CLA
  def _classify(self, label=None):
    """
    Get the labels from the classifier for the last input
    """
    temporalMemoryRegion = self.network.regions["TM"]
    classifierRegion =  self.network.regions["classifier"]

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
      recordNum=self.sequenceId, patternNZ=patternNZ,
      classification=classificationIn)

    return clResults[int(classifierRegion.getParameter("steps"))]


  def testModel(self, numLabels=3):
    """
    Test the kNN classifier on the input sample. Returns the classification most
    frequent amongst the classifications of the sample's individual tokens.
    We ignore the terms that are unclassified, picking the most frequent
    classification among those that are detected.

    @param numLabels      (int)           Number of predicted classifications.
    @return               (numpy array)   The numLabels most-frequent
                                          classifications for the data samples;
                                          values are int or empty.
    """
    sensorRegion = self.network.regions["sensor"]
    spatialPoolerRegion = net.regions["SP"]
    temporalMemoryRegion = self.network.regions["TM"]
    spatialPoolerRegion.setParameter("learningMode", False)
    temporalMemoryRegion.setParameter("learningMode", False)

    self.network.run(1)

    inferredValue = self._classify()
    reset = sensorRegion.getOutputData("categoryOut")[0]

    i = 1 # Hard coded for equal weighting. use lengthOfCurrentSequence later
    if reset:
      self.oldClassifications = numpy.array(inferredValue)
      self.lengthOfCurrentSequence = 1
    else:
      self.lengthOfCurrentSequence += 1
      self.oldClassifications += (numpy.array(inferredValue) * i)

    orderedInferrredValues = sorted(enumerate(self.oldClassifications),
      key=lambda x: x[1], reverse=True)

    labels = zip(*orderedInferredValues)[0]
    return numpy.array(labels[:numLabels])
