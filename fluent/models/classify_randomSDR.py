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

from fluent.models.classification_model import ClassificationModel
from nupic.algorithms.KNNClassifier import KNNClassifier



class ClassificationModelRandomSDR(ClassificationModel):
  """
  Class to run the survey response classification task with random SDRs.
  """

  def __init__(self, verbosity=1):
    super(ClassificationModelRandomSDR, self).__init__(verbosity)

    # Init kNN classifier:
    #   specify 'distanceMethod'='rawOverlap' for overlap; Euclidean is std.
    #   verbosity=1 for debugging
    #   standard k is 1
    self.classifier = KNNClassifier(exact=True, verbosity=verbosity-1)

    # SDR dimensions:
    self.n = 100
    self.w = 20


  def encodePattern(self, string):
    """
    Randomly encode an SDR of the input string, w/ same dimensions
    We seed the random number generator such that a given
    string will yield the same SDR each time this function is called.
    Saving the internal state of the generator reduces the likelihood of
    repeating values from previous inputs.

    @param string     (str)             Text to encode.
    @return           (numpy.array)     Bitmap of the SDR, as a numpy array of
                                        integers.
    """
    state = random.getstate()
    random.seed(string)
    bitmap = numpy.array(random.sample(xrange(self.n), self.w), dtype=int)
    random.setstate(state)
    return numpy.sort(bitmap)


  def resetModel(self):
    """Reset the model by clearing the classifier."""
    self.classifier.clear()


  def trainModel(self, sample, label):
    """
    Train the classifier on the input sample and label.

    @param sample     (list)            List of bitmaps, each representing the
                                        encoding of one token in the sample.
    @param label      (int)             Reference index for the classification
                                        of this sample.
    """
    # This experiment classifies individual tokens w/in each sample. Train the
    # kNN classifier on each token.
    p = 0
    for bitmap in sample:
      if bitmap == []: continue
      p = self.classifier.learn(bitmap, label, isSparse=self.n)


  def testModel(self, sample):
    """
    Test the kNN classifier on the input sample. Returns the classification most
    frequent amongst the classifications of the sample's individual tokens.
    We ignore the terms that are unclassified, picking the most frequent
    classification among those that are detected.
    @param sample           (list)        List of bitmaps, each representing the
                                          encoding of one token in the sample.
    @return classification  (list)        The n most-frequent classifications
                                          for the data samples; for more, see
                                          the KNNClassifier.infer()
                                          documentation. Values are int or None.
    Note: to return multiple winner classifications, modify the return statement
    accordingly.
    """
    tokenLabels = []
    for bitmap in sample:
      if bitmap == []: continue
      (tokenLabel, _, _, _) = self.classifier.infer(
        self._densifyPattern(bitmap))
      if tokenLabel != None:
        # Only include classified tokens.
        tokenLabels.append(tokenLabel)  ## TODO: consider using numpy array (preallocated to len(samples)) for more efficiency
    if tokenLabels == []:
      return [None]
    return self._winningLabels(tokenLabels, n=2)
