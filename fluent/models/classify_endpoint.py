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
import os

from fluent.models.classification_model import ClassificationModel
from fluent.encoders.cio_encoder import CioEncoder

from cortipy.cortical_client import CorticalClient



class ClassificationModelEndpoint(ClassificationModel):
  """
  Class to run the survey response classification task with Cortical.io
  text endpoint encodings and classification.

  From the experiment runner, the methods expect to be fed one sample at a time.
  """

  def __init__(self, verbosity=1):
    super(ClassificationModelEndpoint, self).__init__(verbosity)

    # Init CorticalClient and Cortical.io encoder; need valid API key (see
    # CioEncoder init for details).
    self.encoder = CioEncoder(cacheDir="./experiments/cache")
    self.client = CorticalClient(self.encoder.apiKey)

    self.n = self.encoder.n
    self.w = int((self.encoder.targetSparsity/100) * self.n)

    self.positives = {}
    self.categoryBitmaps = {}


  def encodePattern(self, sample):
    """
    Encode an SDR of the input string by querying the Cortical.io API.

    @param sample     (string)          Original string
    @return           (dictionary)      Dictionary, containing text, sparsity, and bitmap
    """
    text = " ".join(sample)
    fpInfo = self.encoder.encode(text)
    if self.verbosity > 1:
      print "Fingerprint sparsity = {0}%.".format(fpInfo["sparsity"])
    if fpInfo:
      bitmap = numpy.array(fpInfo["fingerprint"]["positions"], dtype="uint32")
      return {"text": text, "sparsity": fpInfo["sparsity"], "bitmap": bitmap}
    else:
      bitmap = numpy.empty(0)
      return {"text": text, "sparsity": float(self.w)/self.n, "bitmap": bitmap}


  def resetModel(self):
    """Reset the model"""
    self.positives.clear()
    self.categoryBitmaps.clear()


  def trainModel(self, sample, label):
    """
    Train the classifier on the input sample and label.

    @param sample     (dictionary)      Dictionary, containing text, sparsity, and bitmap
    @param label      (int)             Reference index for the classification
                                        of this sample.
    """
    if label not in self.positives:
      self.positives[label] = []
    self.positives[label].append(sample["text"])

    categoryBitmap = self.client.createClassification(str(label), self.positives[label])["positions"]

    self.categoryBitmaps[label] = categoryBitmap


  def testModel(self, sample):
    """
    Test the kNN classifier on the input sample. Returns the classification most
    frequent amongst the classifications of the sample's individual tokens.
    We ignore the terms that are unclassified, picking the most frequent
    classification among those that are detected.

    @param sample     (dictionary)      Dictionary, containing text, sparsity, and bitmap
    @return           (dictionary)      The distances between the sample and the classes
    """

    sampleBitmap = sample["bitmap"].tolist()

    distances = {}
    for cat, catBitmap in self.categoryBitmaps.iteritems():
      distances[cat] = self.client.compare(sampleBitmap, catBitmap)

    return distances
