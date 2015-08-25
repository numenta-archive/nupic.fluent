# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import copy
import os

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

  def __init__(self,
               n=100,
               w=20,
               verbosity=1,
               numLabels=3,
               modelDir="ClassificationModelKeywords"):

    super(ClassificationModelKeywords, self).__init__(
        n, w, verbosity=verbosity, numLabels=numLabels, modelDir=modelDir)

    self.classifier = KNNClassifier(exact=True,
                                    distanceMethod="rawOverlap",
                                    k=numLabels,
                                    verbosity=verbosity-1)


  def encodeSample(self, sample):
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
      patterns.append({"text":token,
                       "sparsity":float(self.w)/self.n,
                       "bitmap":self.encodeRandomly(token)})
    return patterns


  def writeOutEncodings(self):
    """
    Log the encoding dictionaries to a txt file; overrides the superclass
    implementation.
    """
    if not os.path.isdir(self.modelDir):
      raise ValueError("Invalid path to write file.")

    # Cast numpy arrays to list objects for serialization.
    jsonPatterns = copy.deepcopy(self.patterns)
    for jp in jsonPatterns:
      for tokenPattern in jp["pattern"]:
        tokenPattern["bitmap"] = tokenPattern.get("bitmap", None).tolist()
      jp["labels"] = jp.get("labels", None).tolist()

    with open(os.path.join(self.modelDir, "encoding_log.txt"), "w") as f:
      f.write(json.dumps(jsonPatterns, indent=1))


  def trainModel(self, i):
    """
    Train the classifier on the sample and labels for record i. The list
    sampleReference is populated to correlate classifier prototypes to sample
    IDs. This model is unique in that a single sample contains multiple encoded
    patterns.

    TODO: add batch training, where i is a list
    """
    for token in self.patterns[i]["pattern"]:
      if token["bitmap"].any():
        for label in self.patterns[i]["labels"]:
          self.classifier.learn(token["bitmap"], label, isSparse=self.n)
          self.sampleReference.append(self.patterns[i]["ID"])


  def testModel(self, i, numLabels=3):
    """
    Test the model on record i.  Returns the classifications
    most frequent amongst the classifications of the sample's individual tokens.
    We ignore the terms that are unclassified, picking the most frequent
    classifications among those that are detected.

    @param numLabels  (int)           Number of classification predictions.
    @return           (numpy array)   numLabels most-frequent classifications
                                      for the data samples; int or empty.
    """
    totalInferenceResult = None
    for pattern in self.patterns[i]["pattern"]:
      if not pattern:
        continue

      (_, inferenceResult, _, _) = self.classifier.infer(
          self._sparsifyPattern(pattern["bitmap"]))

      if totalInferenceResult is None:
        totalInferenceResult = inferenceResult
      else:
        totalInferenceResult += inferenceResult

    return self.getWinningLabels(totalInferenceResult, numLabels)
