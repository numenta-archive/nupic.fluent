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

import unittest
import numpy
import pandas

from fluent.models import classify_random_sdr, classify_endpoint, \
                    classify_fingerprint, classification_model

# Unit tests to add:
# * test `evaluateResults` on an artificial classifications output
# * test `encodePattern` for each model

class ClassificationModelTest(unittest.TestCase):
  """Test the functionality of the classification models."""

  def testClassificationModelTopLabels(self):
    """ Tests whether classification base class returns multiple labels
        correctly.
    """
    model = classification_model.ClassificationModel()
    inferenceResult = numpy.array([3, 1, 4, 0])

    topLabels = model._getTopLabels(inferenceResult, numLabels=1)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2])))

    topLabels = model._getTopLabels(inferenceResult, numLabels=2)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2,0])))

    topLabels = model._getTopLabels(inferenceResult, numLabels=4)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2,0,1,3])))

  def testClassificationModelEvalResultsNone(self):
    """Tests `evaluateResults` method of classification model base class
    checking that accuracy on single label with `None  outputs is as
    expected."""

    model = classification_model.ClassificationModel()

    actualLabels = [[2],[1],[0]]
    predictedLabels = [[None],[2],[None]]
    classifications = [predictedLabels, actualLabels]
    labels = ['cat','fat','sat']

    (accuracy, cm) = model.evaluateResults(classifications, labels, range(3))
    self.assertTrue(numpy.allclose(accuracy, 0.))

  def testClassificationModelEvalResultsSimple(self):
    """Tests `evaluateResults` method of classification model base class
    checking that accuracy on single label outputs is as expected."""

    model = classification_model.ClassificationModel()

    actualLabels = [[2],[1],[0]]
    predictedLabels = [[1],[2],[0]]
    classifications = [predictedLabels, actualLabels]
    labels = ['cat','fat','sat']

    (accuracy, cm) = model.evaluateResults(classifications, labels, range(3))
    self.assertTrue(numpy.allclose(accuracy, 1.0/3))

  def testClassificationModelEvalResultsHard(self):
    """Tests `evaluateResults` method of classification model base class
    checking that accuracy on multiple label outputs is as expected."""

    model = classification_model.ClassificationModel()

    actualLabels = [[2, 1],[1],[0, 1]]
    predictedLabels = [[1],[2],[0]]
    classifications = [predictedLabels, actualLabels]
    labels = ['cat','fat','sat']

    (accuracy, cm) = model.evaluateResults(classifications, labels, range(3))
    self.assertTrue(numpy.allclose(accuracy, 1./3))

  def testClassifyRandomSDRSimple(self):
    """Tests simple classification with single label for `randomSDR`."""
    samples =[['cat'], ['fat']]
    labels = numpy.array([0, 1])
    model = classify_random_sdr.ClassificationModelRandomSDR()

    patterns = []
    [patterns.append(model.encodePattern(samples[idx])) for idx in range(2)]

    for idx, p in enumerate(patterns):
      model.trainModel(p, labels[idx])

    output = [model.testModel(p, 1) for p in patterns]

    self.assertEquals(labels[0], output[0])
    self.assertEquals(labels[1], output[1])

  def testClassifyRandomSDRHard(self):
    """Tests simple classification with multiple labels for `randomSDR`."""
    samples =[['cat'], ['fat']]
    labels = numpy.array([0, 1])
    model = classify_random_sdr.ClassificationModelRandomSDR()

    patterns = []
    [patterns.append(model.encodePattern(samples[idx])) for idx in range(2)]

    for idx, p in enumerate(patterns):
      model.trainModel(p, labels[idx])

    output = [model.testModel(p, 2) for p in patterns]

    self.assertTrue(numpy.allclose(output[0], numpy.array([0,1])))
    self.assertTrue(numpy.allclose(output[1], numpy.array([1,0])))

  @unittest.skip("Ignore tests until its more clear how the Cortical.io "
                 "classifier is performing inference")
  def testClassifyEndpoint(self):
    """Tests sample classification with single label for `endpoint`."""
    samples =[['cat'], ['fat']]
    labels = numpy.array([0, 1])
    model = classify_endpoint.ClassificationModelEndpoint()

    patterns = []
    [patterns.append(model.encodePattern(samples[idx])) for idx in range(2)]

    for idx, p in enumerate(patterns):
      model.trainModel(p, labels[idx])

    output = [model.testModel(p, 1) for p in patterns]

    self.assertTrue(numpy.allclose(output[0], numpy.array([0])))
    self.assertTrue(numpy.allclose(output[1], numpy.array([1])))

  def testClassifyFingerprint(self):
    """Tests sample classification with single label for `fingerprint`."""
    samples =[['cat'], ['fat']]
    labels = numpy.array([0, 1])
    model = classify_fingerprint.ClassificationModelFingerprint()

    patterns = []
    [patterns.append(model.encodePattern(samples[idx])) for idx in range(2)]

    for idx, p in enumerate(patterns):
      model.trainModel(p, labels[idx])

    output = [model.testModel(p, 1) for p in patterns]

    self.assertEquals(labels[0], output[0])
    self.assertEquals(labels[1], output[1])

  def testClassifyFingerprintHard(self):
    """Tests simple classification with multiple labels for `fingerprint`."""
    samples =[['cat'], ['fat']]
    labels = numpy.array([0, 1])
    model = classify_random_sdr.ClassificationModelRandomSDR()

    patterns = []
    [patterns.append(model.encodePattern(samples[idx])) for idx in range(2)]

    for idx, p in enumerate(patterns):
      model.trainModel(p, labels[idx])

    output = [model.testModel(p, 2) for p in patterns]

    self.assertTrue(numpy.allclose(output[0], numpy.array([0,1])))
    self.assertTrue(numpy.allclose(output[1], numpy.array([1,0])))

if __name__ == "__main__":
  unittest.main()
