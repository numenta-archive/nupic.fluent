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

  def __init__(self, verbosity=1):
    """
    Initialize the CorticalClient and CioEncoder. Requires a valid API key
    """
    super(ClassificationModelEndpoint, self).__init__(verbosity)

    self.encoder = CioEncoder(cacheDir="./experiments/cache")
    self.client = CorticalClient(self.encoder.apiKey)

    self.n = self.encoder.n
    self.w = int((self.encoder.targetSparsity/100) * self.n)

    self.positives = {}
    self.negatives = {}
    self.categoryBitmaps = {}


  def encodePattern(self, pattern):
    """
    Encode an SDR of the input string by querying the Cortical.io API.

    @param pattern        (list)          Tokenized sample, where each item is
                                          a string
    @return               (dict)          The sample text, sparsity, and bitmap.
    Example return dict:
    {
      "text": "Example text",
      "sparsity": 0.03,
      "bitmap": numpy.array()
    }
    """
    text = " ".join(pattern)
    fpInfo = self.encoder.encode(text)
    if self.verbosity > 1:
      print "Fingerprint sparsity = {0}%.".format(fpInfo["sparsity"])

    if fpInfo:
      text = fpInfo["text"] if "text" in fpInfo else fpInfo["term"]
      bitmap = numpy.array(fpInfo["fingerprint"]["positions"])
      sparsity = fpInfo["sparsity"]
    else:
      bitmap = self.encodeRandomly(text)
      sparsity = float(self.w) / self.n

    return {"text": text, "sparsity": sparsity, "bitmap": bitmap}


  def resetModel(self):
    """Reset the model"""
    self.positives.clear()
    self.negatives.clear()
    self.categoryBitmaps.clear()


  def trainModel(self, sample, label, negatives=None):
    """
    Train the classifier on the input sample and label. Use Cortical.io's
    createClassification to make a bitmap that represents the class

    @param sample     (dict)            The sample text, sparsity, and bitmap.
    @param label      (int)             Reference index for the classification
                                        of this sample.
    @param negatives  (list)            Each item is the dictionary containing
                                        text, sparsity and bitmap for the
                                        negative samples
    """
    if label not in self.positives:
      self.positives[label] = []
    
    if sample["text"] != "":
      self.positives[label].append(sample["text"])

    if label not in self.negatives:
      self.negatives[label] = []

    if negatives is not None:
      for neg in negatives:
        if neg["text"] != "":
          self.negatives[label].append(neg["text"])

    self.categoryBitmaps[label] = self.client.createClassification(str(label),
      self.positives[label], self.negatives[label])["positions"]


  def testModel(self, sample, numberCats=1, metric="overlappingAll"):
    """
    Test the Cortical.io classifier on the input sample. Returns a dictionary
    containing various distance metrics between the sample and the classes.

    @param sample         (dict)      The sample text, sparsity, and bitmap.
    @return               (list)      Winning classifications based on the
                                      specified metric. The number of items
                                      returned will be <= numberCats.
    """
    sampleBitmap = sample["bitmap"].tolist()

    distances = defaultdict(list)
    for cat, catBitmap in self.categoryBitmaps.iteritems():
      distances[cat] = self.client.compare(sampleBitmap, catBitmap)

    return self.winningLabels(distances, numberCats=1, metric="overlappingAll")  ## TODO: how do we handle return values of []? or len(winners)<numberCats?


  @staticmethod
  def winningLabels(distances, numberCats, metric):
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

    return [distances.keys()[catIdx] for catIdx in sortedIdx[:numberCats]]
