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
from fluent.encoders import EncoderTypes
from fluent.models.classification_model import ClassificationModel
from nupic.algorithms.KNNClassifier import KNNClassifier



class ClassificationModelFingerprint(ClassificationModel):
  """
  Class to run the survey response classification task with Coritcal.io
  fingerprint encodings.

  From the experiment runner, the methods expect to be fed one sample at a time.
  """

  def __init__(self,
               verbosity=1,
               numLabels=3,
               fingerprintType=EncoderTypes.word):

    super(ClassificationModelFingerprint, self).__init__(verbosity, numLabels)

    # Init kNN classifier and Cortical.io encoder; need valid API key (see
    # CioEncoder init for details).
    self.classifier = KNNClassifier(k=numLabels,
                                    distanceMethod='rawOverlap',
                                    exact=False,
                                    verbosity=verbosity-1)

    if fingerprintType is (not EncoderTypes.document or not EncoderTypes.word):
      raise ValueError("Invaid type of fingerprint encoding; see the "
                       "EncoderTypes class for eligble types.")
    self.encoder = CioEncoder(cacheDir="./fluent/experiments/cioCache",
                              fingerprintType=fingerprintType)
    self.n = self.encoder.n
    self.w = int((self.encoder.targetSparsity/100)*self.n)


  def encodePattern(self, sample):
    """
    Encode an SDR of the input string by querying the Cortical.io API. If the
    client returns None, we create a random SDR with the model's dimensions n
    and w.

    @param sample     (list)            Tokenized sample, where each item is a
                                        string token.
    @return fp        (dict)            The sample text, sparsity, and bitmap.
    Example return dict:
      {
        "text": "Example text",
        "sparsity": 0.03,
        "bitmap": numpy.array([])
      }
    """
    sample = " ".join(sample)
    fpInfo = self.encoder.encode(sample)
    if fpInfo:
      fp = {"text":fpInfo["text"] if "text" in fpInfo else fpInfo["term"],
            "sparsity":fpInfo["sparsity"],
            "bitmap":numpy.array(fpInfo["fingerprint"]["positions"])
            }
    else:
      fp = {"text":sample,
            "sparsity":float(self.w)/self.n,
            "bitmap":self.encodeRandomly(sample)
            }

    return fp


  def resetModel(self):
    """Reset the model by clearing the classifier."""
    self.classifier.clear()


  def trainModel(self, samples, labels):
    """
    Train the classifier on the input sample and labels.

    @param samples    (list)          List of dictionaries containing the
                                      sample text, sparsity, and bitmap.
    @param labels     (list)          List of numpy arrays containing the
                                      reference indices for the classifications
                                      of each sample.
    """
    for sample, sample_labels in zip(samples, labels):
      if sample["bitmap"].any():
        for label in sample_labels:
          self.classifier.learn(sample["bitmap"], label, isSparse=self.n)


  def testModel(self, sample, numLabels=3):
    """
    Test the kNN classifier on the input sample. Returns the classification most
    frequent amongst the classifications of the sample's individual tokens.
    We ignore the terms that are unclassified, picking the most frequent
    classification among those that are detected.

    @param sample         (dict)          The sample text, sparsity, and bitmap.
    @param numLabels      (int)           Number of predicted classifications.
    @return               (numpy array)   The numLabels most-frequent
                                          classifications for the data samples;
                                          values are int or empty.
    """
    (_, inferenceResult, _, _) = self.classifier.infer(
      self._densifyPattern(sample["bitmap"]))
    return self.getWinningLabels(inferenceResult, numLabels)
