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
import pandas
import unittest

from fluent.models.classification_model import ClassificationModel
from fluent.models.classify_endpoint import ClassificationModelEndpoint
from fluent.models.classify_fingerprint import ClassificationModelFingerprint
from fluent.models.classify_random_sdr import ClassificationModelRandomSDR



class ClassificationModelTest(unittest.TestCase):
  """Test the functionality of the classification models."""

  def testClassificationModelTopLabels(self):
    """ Tests whether classification base class returns multiple labels
        correctly.
    """
    model = ClassificationModel()
    inferenceResult = numpy.array([3, 1, 4, 0])

    topLabels = model._getTopLabels(inferenceResult, numLabels=1)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2])),
                    "Output labels do not match what is expected.")

    topLabels = model._getTopLabels(inferenceResult, numLabels=2)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2,0])),
                    "Output labels do not match what is expected.")

    topLabels = model._getTopLabels(inferenceResult, numLabels=4)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2,0,1,3])),
                    "Output labels do not match what is expected.")


  def testClassificationModelEvalResultsNone(self):
    """Tests `evaluateResults` method of classification model base class
    checking that accuracy on single label with `None  outputs is as
    expected."""

    model = ClassificationModel()

    actualLabels = [[2],[1],[0]]
    predictedLabels = [[None],[2],[None]]
    classifications = [predictedLabels, actualLabels]
    labels = ['cat','fat','sat']

    (accuracy, cm) = model.evaluateResults(classifications, labels, xrange(3))
    self.assertTrue(numpy.allclose(accuracy, 0.),
                    "Output accuracy does not match what is expected.")


  def testClassificationModelEvalResultsSimple(self):
    """Tests `evaluateResults` method of classification model base class
    checking that accuracy on single label outputs is as expected."""

    model = ClassificationModel()

    actualLabels = [[2],[1],[0]]
    predictedLabels = [[1],[2],[0]]
    classifications = [predictedLabels, actualLabels]
    labels = ['cat','fat','sat']

    (accuracy, cm) = model.evaluateResults(classifications, labels, xrange(3))
    self.assertTrue(numpy.allclose(accuracy, 1.0/3),
                    "Output accuracy does not match what is expected.")


  def testClassificationModelEvalResultsHard(self):
    """Tests `evaluateResults` method of classification model base class
    checking that accuracy on multiple label outputs is as expected."""

    model = ClassificationModel()

    actualLabels = [[2, 1],[1],[0, 1]]
    predictedLabels = [[1],[2],[0]]
    classifications = [predictedLabels, actualLabels]
    labels = ['cat','fat','sat']

    (accuracy, cm) = model.evaluateResults(classifications, labels, xrange(3))
    self.assertTrue(numpy.allclose(accuracy, 1./3),
                    "Output accuracy does not match what is expected.")


  def testClassifyRandomSDRSimple(self):
    """Tests simple classification with single label for `randomSDR`."""
    samples =[['cat'], ['fat']]
    labels = numpy.array([0, 1])
    model = ClassificationModelRandomSDR()

    patterns = []
    [patterns.append(model.encodePattern(samples[idx])) for idx in xrange(2)]

    for label, pattern in zip(labels, patterns):
      model.trainModel(pattern, label)

    output = [model.testModel(p, 1) for p in patterns]

    self.assertEquals(labels[0], output[0], "Output labels do not match what "
                                            "is expected.")
    self.assertEquals(labels[1], output[1], "Output labels do not match what "
                                            "is expected.")


  def testClassifyRandomSDRHard(self):
    """Tests simple classification with multiple labels for `randomSDR`."""
    samples =[['cat'], ['fat']]
    labels = numpy.array([0, 1])
    model = ClassificationModelRandomSDR()

    patterns = [model.encodePattern(samples[idx]) for idx in xrange(2)]

    for label, pattern in zip(labels, patterns):
      model.trainModel(pattern, label)

    output = [model.testModel(p, 2) for p in patterns]

    self.assertTrue(numpy.allclose(output[0], numpy.array([0,1])),
                    "Output labels do not match what is expected.")
    self.assertTrue(numpy.allclose(output[1], numpy.array([1,0])),
                    "Output labels do not match what is expected.")


  @unittest.skip("Ignore tests until its more clear how the Cortical.io "
                 "classifier is performing inference")
  def testClassifyEndpoint(self):
    """Tests sample classification with single label for `endpoint`."""
    samples =[['cat'], ['fat']]
    labels = numpy.array([0, 1])
    model = ClassificationModelEndpoint()

    patterns = []
    [patterns.append(model.encodePattern(samples[idx])) for idx in xrange(2)]

    for idx, p in enumerate(patterns):
      model.trainModel(p, labels[idx])

    output = [model.testModel(p, 1) for p in patterns]

    self.assertTrue(numpy.allclose(output[0], numpy.array([0])),
                    "Output labels do not match what is expected.")
    self.assertTrue(numpy.allclose(output[1], numpy.array([1])),
                    "Output labels do not match what is expected.")


  def testClassifyFingerprint(self):
    """Tests sample classification with single label for `fingerprint`."""
    samples =[['cat'], ['fat']]
    labels = numpy.array([0, 1])
    model = ClassificationModelFingerprint()

    patterns = []
    [patterns.append(model.encodePattern(samples[idx])) for idx in xrange(2)]

    for label, pattern in zip(labels, patterns):
      model.trainModel(pattern, label)

    output = [model.testModel(p, 1) for p in patterns]

    self.assertEquals(labels[0], output[0],
                      "Output labels do not match what is expected.")
    self.assertEquals(labels[1], output[1],
                      "Output labels do not match what is expected.")


  def testClassifyFingerprintHard(self):
    """Tests simple classification with multiple labels for `fingerprint`."""
    samples =[['cat'], ['fat']]
    labels = numpy.array([0, 1])
    model = ClassificationModelRandomSDR()

    patterns = []
    [patterns.append(model.encodePattern(samples[idx])) for idx in xrange(2)]

    for label, pattern in zip(labels, patterns):
      model.trainModel(pattern, label)

    output = [model.testModel(p, 2) for p in patterns]

    self.assertTrue(numpy.allclose(output[0], numpy.array([0,1])),
                    "Output labels do not match what is expected.")
    self.assertTrue(numpy.allclose(output[1], numpy.array([1,0])),
                    "Output labels do not match what is expected.")


if __name__ == "__main__":
  unittest.main()
