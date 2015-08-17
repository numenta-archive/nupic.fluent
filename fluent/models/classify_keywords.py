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

import copy
import numpy
import os
import random

from fluent.models.classification_model import ClassificationModel
from nupic.algorithms.KNNClassifier import KNNClassifier

try:
  import simplejson as json
except ImportError:
  import json



class ClassificationModelKeywords(ClassificationModel):
  """
  Class to run the survey response classification task with random SDRs.

  From the experiment runner, the methods expect to be fed one sample at a time.

  TODO: use nupic.bindings.math import Random
  """

  def __init__(self, n=100, w=20, verbosity=1, numLabels=3):
    super(ClassificationModelKeywords, self).__init__(n, w, verbosity,
                                                       numLabels)

    self.classifier = KNNClassifier(exact=True,
                                    distanceMethod='rawOverlap',
                                    k=numLabels,
                                    verbosity=verbosity-1)


  def encodePattern(self, sample):
    """
    Randomly encode an SDR of the input strings. We seed the random number
    generator such that a given string will yield the same SDR each time this
    method is called.

    @param sample     (list)            Tokenized sample, where each item is a
                                        string token.
    @return           (list)            Numpy arrays, each with a bitmap of the
                                        encoding.
    """
    patterns = []
    for token in sample:
      patterns.append({
                        "text":token,
                        "sparsity":float(self.w)/self.n,
                        "bitmap":self.encodeRandomly(token)
                        })
    return patterns


  def writeOutEncodings(self, patterns, path):
    """
    Log the encoding dictionaries to a txt file; overrides the superclass
    implementation.
    """
    if not os.path.isdir(path):
      raise ValueError("Invalid path to write file.")

    # Cast numpy arrays to list objects for serialization.
    jsonPatterns = copy.deepcopy(patterns)
    for jp in jsonPatterns:
      for tokenPattern in jp["pattern"]:
        tokenPattern["bitmap"] = tokenPattern.get("bitmap", None).tolist()
      jp["labels"] = jp.get("labels", None).tolist()

    with open(os.path.join(path, "encoding_log.txt"), "w") as f:
      f.write(json.dumps(jsonPatterns, indent=1))


  def resetModel(self):
    """Reset the model by clearing the classifier."""
    self.classifier.clear()


  def trainModel(self, samples, labels):
    """
    Train the classifier on the input sample and label. This model is unique in
    that a single sample contains multiple encoded patterns.

    @param samples    (list)          List of list of dicts, each representing
                                      the encoding of one token in a sample.
    @param labels     (list)          List of numpy arrays containing the
                                      reference indices for the classifications
                                      of each sample.
    """
    # This experiment classifies individual tokens w/in each sample. Train the
    # classifier on each token.
    for sample, sample_labels in zip(samples, labels):
      for token in sample:
        if not token: continue
        for label in sample_labels:
          self.classifier.learn(token["bitmap"], label, isSparse=self.n)


  def testModel(self, sample, numLabels=3):
    """
    Test the classifier on the input sample. Returns the classifications
    most frequent amongst the classifications of the sample's individual tokens.
    We ignore the terms that are unclassified, picking the most frequent
    classifications among those that are detected.
    @param sample           (list)          List of dict encodings, one for each
                                            token in the sample.
    @param numLabels        (int)           Number of predicted classifications.
    @return                 (numpy array)   The numLabels most-frequent
                                            classifications for the data
                                            samples; values are int or empty.
    """
    totalInferenceResult = None
    for idx, s in enumerate(sample):
      if not s: continue

      (_, inferenceResult, _, _) = self.classifier.infer(
        self._densifyPattern(s["bitmap"]))

      if totalInferenceResult is None:
        totalInferenceResult = inferenceResult
      else:
        totalInferenceResult += inferenceResult

    return self.getWinningLabels(totalInferenceResult, numLabels)
