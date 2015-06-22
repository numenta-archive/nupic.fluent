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
import random
import string

from fluent.encoders.cio_encoder import CioEncoder
from fluent.models.classification_model import ClassificationModel
from nupic.algorithms.KNNClassifier import KNNClassifier



class ClassificationModelFingerprint(ClassificationModel):
  """
  Class to run the survey response classification task with Coritcal.io
  fingerprint encodings.

  From the experiment runner, the methods expect to be fed one sample at a time.
  """

  def __init__(self, verbosity=1):
    super(ClassificationModelFingerprint, self).__init__(verbosity)

    # Init kNN classifier and Cortical.io encoder; need valid API key (see
    # CioEncoder init for details).
    self.classifier = KNNClassifier(k=1, exact=False, verbosity=verbosity-1)
    self.encoder = CioEncoder(cacheDir="./experiments/cache")
    self.n = self.encoder.n
    self.w = int((self.encoder.targetSparsity/100)*self.n)


  def encodePattern(self, sample):
    """
    Encode an SDR of the input string by querying the Cortical.io API.

    @param sample     (list)            Tokenized sample, where each item is a
                                        string token.
    @return           (list)            Numpy arrays, each with a bitmap of the
                                        encoding.
    """
    fpInfo = self.encoder.encode(string.join(sample))
    if self.verbosity > 1:
      print "Fingerprint sparsity = {0}%.".format(fpInfo["sparsity"])
    if fpInfo:
      return numpy.array(fpInfo["fingerprint"]["positions"], dtype="uint32")
    else:
      return numpy.empty(0)


  def resetModel(self):
    """Reset the model by clearing the classifier."""
    self.classifier.clear()


  def trainModel(self, sample, label):
    """
    Train the classifier on the input sample and label.

    @param sample     (numpy.array)     Bitmap encoding of the sample.
    @param label      (int)             Reference index for the classification
                                        of this sample.
    """
    if sample.any():
      _ = self.classifier.learn(sample, label, isSparse=self.n)


  def testModel(self, sample):
    """
    Test the kNN classifier on the input sample. Returns the classification most
    frequent amongst the classifications of the sample's individual tokens.
    We ignore the terms that are unclassified, picking the most frequent
    classification among those that are detected.

    @param sample     (numpy.array)     Bitmap encoding of the sample.
    @return           (list)            The n most-frequent classifications
                                        for the data samples; for more, see the
                                        KNNClassifier.infer() documentation.
                                        Values are int or None.
    Note: to return multiple winner classifications, modify the return statement
    accordingly.
    """
    tokenLabels = []
    (tokenLabel, _, _, _) = self.classifier.infer(self._densifyPattern(sample))
    ## TODO: get list of closest classifications, not just the winner
    return [tokenLabel]
