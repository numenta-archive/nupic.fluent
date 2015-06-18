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

from fluent.encoders.cio_encoder import CioEncoder
from fluent.models.classification_model import ClassificationModel

from cortipy.cortical_client import CorticalClient



class ClassificationModelEndpoint(ClassificationModel):
  """
  Class to run the survey response classification task with Coritcal.io
  endpoint encodings.

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
    self.categorySDRs = {}


  def encodePattern(self, sample):
    """
    Encode an SDR of the input string by querying the Cortical.io API.

    @param sample     (list)            Tokenized sample, where each item is a
                                        string token.
    @return           (list)            Numpy arrays, each with a bitmap of the
                                        encoding.
    """
    fpInfo = self.encoder.encode(" ".join(sample))
    if self.verbosity > 1:
      print "Fingerprint sparsity = {0}%.".format(fpInfo["sparsity"])
    if fpInfo:
      return numpy.array(fpInfo["fingerprint"]["positions"], dtype="uint32")
    else:
      return numpy.empty(0)


  def resetModel(self):
    """Reset the model"""
    self.positives.clear()
    self.categorySDRs.clear()


  def trainModel(self, sample, label):
    """
    Train the classifier on the input sample and label.

    @param sample     (string)          Comment
    @param label      (int)             Reference index for the classification
                                        of this sample.
    """
    if label not in self.positives:
      self.positives[label] = []
    self.positives[label].append(sample)

    categoryBitmap = self.client.createClassification(label, self.positives[label])["positions"]

    self.categorySDRs[label] = self.densifyPattern(categoryBitmap)


  def testModel(self, sample):
    """
    Test the kNN classifier on the input sample. Returns the classification most
    frequent amongst the classifications of the sample's individual tokens.
    We ignore the terms that are unclassified, picking the most frequent
    classification among those that are detected.

    @param sample     (numpy.array)     bitmap encoding of the sample.
    @return           (dictionary)      The distances between the sample and the classes
    """

    sampleSDR = self.densifyPattern(sample)

    distances = {}
    for cat, catSDR in self.categorySDRs.iteritems():
      distances[cat] = self.encoder.compare(catSDR, sampleSDR)

    return distances
