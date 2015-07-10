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

from collections import defaultdict
from cortipy.cortical_client import CorticalClient
from fluent.encoders.cio_encoder import CioEncoder
from fluent.models.classification_model import ClassificationModel



class ClassificationModelEndpoint(ClassificationModel):
  """
  Class to run the survey response classification task with Cortical.io
  text endpoint encodings and classification system.

  From the experiment runner, the methods expect to be fed one sample at a time.
  """

  def __init__(self, verbosity=1, multiclass=False):
    """
    Initialize the CorticalClient and CioEncoder. Requires a valid API key
    """
    super(ClassificationModelEndpoint, self).__init__(verbosity, multiclass)

    self.encoder = CioEncoder(cacheDir="./experiments/cache")
    self.client = CorticalClient(self.encoder.apiKey)

    self.n = self.encoder.n
    self.w = int((self.encoder.targetSparsity/100) * self.n)

    self.categoryBitmaps = {}
    self.negatives = {}
    self.positives = {}


  def encodePattern(self, sample):
    """
    Encode an SDR of the input string by querying the Cortical.io API.

    @param sample         (list)          Tokenized sample, where each item is
                                          a string
    @return fp            (dict)          The sample text, sparsity, and bitmap.
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
    """Reset the model"""
    self.positives.clear()
    self.negatives.clear()
    self.categoryBitmaps.clear()


  ## TODO: move Cortical.io client logic to CioEncoder.
  def trainModel(self, sample, labels, negatives=None):
    """
    Train the classifier on the input sample and label. Use Cortical.io's
    createClassification to make a bitmap that represents the class

    @param sample     (dict)            The sample text, sparsity, and bitmap.
    @param labels     (numpy array)     Reference indices for the
                                        classifications of this sample.
    @param negatives  (list)            Each item is the dictionary containing
                                        text, sparsity and bitmap for the
                                        negative samples.
    """
    for label in labels:
      if label not in self.positives:
        self.positives[label] = []

      if sample["text"]:
        self.positives[label].append(sample["text"])

      if label not in self.negatives:
        self.negatives[label] = []

      if negatives:
        for neg in negatives:
          if neg["text"]:
            self.negatives[label].append(neg["text"])

      self.categoryBitmaps[label] = self.client.createClassification(
          str(label),
          self.positives[label],
          self.negatives[label])["positions"]


  def testModel(self, sample, numLabels=3, metric="overlappingAll"):
    """
    Test the Cortical.io classifier on the input sample. Returns a dictionary
    containing various distance metrics between the sample and the classes.

    @param sample         (dict)      The sample text, sparsity, and bitmap.
    @return               (list)      Winning classifications based on the
                                      specified metric. The number of items
                                      returned will be <= numLabels.
    """
    sampleBitmap = sample["bitmap"].tolist()

    distances = defaultdict(list)
    for cat, catBitmap in self.categoryBitmaps.iteritems():
      distances[cat] = self.client.compare(sampleBitmap, catBitmap)

    return self.getWinningLabels(distances, numLabels=numLabels, metric=metric)


  @staticmethod
  def getWinningLabels(distances, numLabels, metric):
    """
    Return indices of winning categories, based off of the input metric.
    Overrides the base class implementation.
    """
    metricValues = numpy.array([v[metric] for v in distances.values()])
    sortedIdx = numpy.argsort(metricValues)

    # euclideanDistance and jaccardDistance are ascending
    descendingOrder = set(["overlappingAll", "overlappingLeftRight",
      "overlappingRightLeft", "cosineSimilarity", "weightedScoring"])
    if metric in descendingOrder:
      sortedIdx = sortedIdx[::-1]

    return numpy.array(
        [distances.keys()[catIdx] for catIdx in sortedIdx[:numLabels]])
