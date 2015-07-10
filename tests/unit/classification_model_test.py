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

  def testWinningLabels(self):
    """
    Tests whether classification base class returns multiple labels correctly.
    """
    model = ClassificationModel()
    inferenceResult = numpy.array([3, 1, 4, 0, 1, 0])

    topLabels = model.getWinningLabels(inferenceResult, numLabels=1)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2])),
                    "Output should be label 2.")

    topLabels = model.getWinningLabels(inferenceResult, numLabels=2)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2, 0])),
                    "Output should be labels 2 and 0.")

    # Test tie is broken by the later index; label 4 beats out label 1.
    topLabels = model.getWinningLabels(inferenceResult, numLabels=3)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2, 0, 4])),
                    "Output should be labels 2, 0, and 1.")

    # Test only nonzero labels are returned.
    topLabels = model.getWinningLabels(inferenceResult, numLabels=5)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2, 0, 4, 1])),
                    "Output should be labels 2, 0, 1, and 4.")


  def testNoWinningLabels(self):
    """Inferring 0/4 classes should return 0 winning labels."""
    model = ClassificationModel()

    inferenceResult = numpy.array([0, 0, 0, 0])
    topLabels = model.getWinningLabels(inferenceResult)

    self.assertFalse(topLabels)


  def testCalculateAccuracyMixedSamples(self):
    """
    Tests testCalculateAccuracy() method of classification model base class for
    test samples with mixed classifications.
    """
    model = ClassificationModel()

    actualLabels = [numpy.array([0, 1, 2])]
    predictedLabels1 = [numpy.array([1, 2, 0])]
    predictedLabels2 = [numpy.array([1])]
    predictedLabels3 = [None]
    classifications1 = [predictedLabels1, actualLabels]
    classifications2 = [predictedLabels2, actualLabels]
    classifications3 = [predictedLabels3, actualLabels]

    self.assertAlmostEqual(model.calculateAccuracy(classifications1), 1.0)
    self.assertAlmostEqual(
        model.calculateAccuracy(classifications2), float(1)/3)
    self.assertAlmostEqual(model.calculateAccuracy(classifications3), 0.0)


  def testCalculateAccuracyMultipleSamples(self):
    """
    Tests testCalculateAccuracy() method of classification model base class for
    three test samples.
    """
    model = ClassificationModel()

    actualLabels = [numpy.array([0]),
                    numpy.array([0, 2]),
                    numpy.array([0, 1, 2])]
    predictedLabels = [numpy.array([0]),
                       [None],
                       numpy.array([1, 2, 0])]
    classifications = [predictedLabels, actualLabels]

    self.assertAlmostEqual(model.calculateAccuracy(classifications), float(2)/3)


  def testClassifyRandomSDRSingleAndMultiClass(self):
    """Tests simple classification with multiple labels for randomSDR model."""
    model = ClassificationModelRandomSDR()

    samples =[(["Pickachu"], numpy.array([0, 2])),
              (["Eevee"], numpy.array([2])),
              (["Charmander"], numpy.array([0, 1])),
              (["Abra"], numpy.array([1])),
              (["Squirtle"], numpy.array([1, 0]))]

    patterns = [{"pattern": model.encodePattern(s[0]),
                 "labels": s[1]}
                for s in samples]

    for i in xrange(len(samples)):
      model.trainModel(patterns[i]["pattern"], patterns[i]["labels"])

    output = [model.testModel(p["pattern"]) for p in patterns]

    self.assertSequenceEqual(output[0].tolist(), [2, 0],
                             "Incorrect output for first sample.")
    self.assertSequenceEqual(output[1].tolist(), [2],
                             "Incorrect output for second sample.")
    self.assertSequenceEqual(output[2].tolist(), [1, 0],
                             "Incorrect output for third sample.")
    self.assertSequenceEqual(output[3].tolist(), [1],
                             "Incorrect output for fourth sample.")

    # Test the order of class labels doesn't matter when training.
    self.assertTrue(numpy.allclose(output[2], output[4]),
                    "Outputs for samples 2 and 4 should be identical.")


## TODO: ClassificationModelEndpoint/Fingerprint tests (mock out encodings)


if __name__ == "__main__":
  unittest.main()
