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

from collections import Counter
from fluent.models.classification_model import ClassificationModel
from nupic.algorithms.KNNClassifier import KNNClassifier



class ClassificationModelRandomSDR(ClassificationModel):
  """
  Class to run the survey response classification task with random SDRs.
  """

  def __init__(self,
      encoder='random',     # in this case, only for model info
  		kCV=3,  						  # if = 1, no cross-validation
      paths={},
      verbosity=1
  	):

    # Unpack params:
    self.encoder      = encoder
    self.kCV          = kCV
    self.paths        = paths
    self.verbosity    = verbosity

    # Init kNN classifier:
    # specify 'distanceMethod'='rawOverlap' for overlap; Euclidean is std.
    # pass verbosity=1 for debugging
    self.classifier = KNNClassifier()  

    # SDR dimensions:
    self.n = 16384
    self.w = 328


  def _winningLabel(self, labels):  ## move up to base
    """Returns the most frequent item in the input list of labels."""
    try:
      data = Counter(labels)
      return data.most_common(1)[0][0]
    except IndexError:
      import pdb; pdb.set_trace()


  def encodePattern(self, string):
    """
    Returns a randomly encoded SDR of the input string, w/ same dimensions 
    as the Cio encoder. We seed the random number generator such that a given 
    string will yield the same SDR each time this function is called.
    Saving the internal state of the generator reduces the likelihood of
    repeating values from previous inputs.
    """
    state = random.getstate()
    random.seed(string)
    bitmap = random.sample(xrange(self.n), self.w)
    random.setstate(state)
    return sorted(bitmap)


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
    for bitmap in sample:
      if bitmap == []: continue
      p = self.classifier.learn(bitmap, label, isSparse=self.n)
      if self.verbosity > 0:
        print "\tcumulative number of patterns classified = %i" % p


  def testModel(self, sample):
    """
    Test the kNN classifier on the input sample. Returns the classification most
    frequent amongst the classifications of the sample's individual tokens.
    @param sample           (list)        List of bitmaps, each representing the
                                          encoding of one token in the sample.
    @return classification  (int)         The 'winner' classifications for the
                                          data samples; for more, see the
                                          KNNClassifier.infer() documentation.
    """
    tokenLabels = []
    for bitmap in sample:
      if bitmap == []: continue
      (tokenLabel, _, _, _) = self.classifier.infer(self.densifyPattern(bitmap))
      tokenLabels.append(tokenLabel)
    if tokenLabels == []: return []
    return self._winningLabel(tokenLabels)
