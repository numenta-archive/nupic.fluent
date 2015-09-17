#!/usr/bin/env python
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

import os
import shutil
import unittest

from fluent.encoders import EncoderTypes
from fluent.experiments.htm_runner import HTMRunner
from fluent.experiments.runner import Runner
from fluent.utils.csv_helper import readCSV


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")



class ClassificationModelsTest(unittest.TestCase):
  """Test class for ClassificationModelKeywords."""

  @staticmethod
  def runExperiment(runner):
    try:
      runner.setupData()
      runner.encodeSamples()
      runner.runExperiment()
    except Exception as e:
      print "Runner could not execute the experiment."
      raise e
    finally:
      # cleanup
      shutil.rmtree(runner.model.modelDir.split("/")[0])
  
  
  @staticmethod
  def getExpectedClassifications(runner, expectationFilePath):
    """
    Return a list of the labels predicted by runner and a list of expected 
    labels from the expected classifications file path.
    """
    dataDict = readCSV(expectationFilePath, numLabels=3)

    expectedClasses = []
    resultClasses = []
    for i, trial in enumerate(runner.results):
      for j, predictionList in enumerate(trial[0]):
        predictions = [runner.labelRefs[p] for p in predictionList]
        if predictions == []:
          predictions = ["(none)"]
        resultClasses.append(predictions)
        expectedClasses.append(dataDict.items()[j+runner.trainSizes[i]][1][1])

    return expectedClasses, resultClasses


  def testClassifyKeywordsAsExpected(self):
    """
    Tests ClassificationModelKeywords.
    
    Training on the first five samples of the dataset, and testing on the rest,
    the model's classifications should match those in the expected classes
    data file.
    """
    modelName = "Keywords"
    runner = Runner(dataPath=os.path.join(DATA_DIR, "responses.csv"),
                    resultsDir="",
                    experimentName="keywords_test",
                    loadPath=None,
                    modelName=modelName,
                    numClasses=3,
                    plots=0,
                    orderedSplit=True,
                    trainSizes=[5],
                    verbosity=0)
    runner.initModel(modelName)
    self.runExperiment(runner)

    expectedClasses, resultClasses = self.getExpectedClassifications(
      runner, os.path.join(DATA_DIR, "responses_expected_classes_keywords.csv"))

    for i, (e, r) in enumerate(zip(expectedClasses, resultClasses)):
      if i in (7, 9, 12):
        # Ties amongst winning labels are handled randomly, which affects the
        # third classification in these test samples.
        e = e[:2]
        r = r[:2]
      self.assertEqual(sorted(e), sorted(r),
      "Keywords model predicted classes other than what we expect.")


  def testClassifyDocumentFingerprintsAsExpected(self):
    """
    Tests ClassificationModelFingerprint (for encoder type 'document').
    
    Training on the first five samples of the dataset, and testing on the rest,
    the model's classifications should match those in the expected classes
    data file.
    """
    modelName = "CioDocumentFingerprint"
    runner = Runner(dataPath=os.path.join(DATA_DIR, "responses.csv"),
                    resultsDir="",
                    experimentName="fingerprints_test",
                    loadPath=None,
                    modelName=modelName,
                    numClasses=3,
                    plots=0,
                    orderedSplit=True,
                    trainSizes=[5],
                    verbosity=0)
    runner.initModel(modelName)
    runner.model.encoder.fingerprintType = EncoderTypes.document
    self.runExperiment(runner)

    expectedClasses, resultClasses = self.getExpectedClassifications(runner,
      os.path.join(DATA_DIR,
                   "responses_expected_classes_fingerprint_document.csv"))

    [self.assertEqual(sorted(e), sorted(r),
      "Fingerprint model predicted classes other than what we expect.")
      for e, r in zip(expectedClasses, resultClasses)]


  def testClassifyWordFingerprintsAsExpected(self):
    """
    Tests ClassificationModelFingerprint (for encoder type 'word').
    
    Training on the first five samples of the dataset, and testing on the rest,
    the model's classifications should match those in the expected classes
    data file.
    """
    modelName = "CioWordFingerprint"
    runner = Runner(dataPath=os.path.join(DATA_DIR, "responses.csv"),
                    resultsDir="",
                    experimentName="fingerprints_test",
                    loadPath=None,
                    modelName=modelName,
                    numClasses=3,
                    plots=0,
                    orderedSplit=True,
                    trainSizes=[5],
                    verbosity=0)
    runner.initModel(modelName)
    runner.model.encoder.fingerprintType = EncoderTypes.word
    self.runExperiment(runner)

    expectedClasses, resultClasses = self.getExpectedClassifications(runner,
      os.path.join(DATA_DIR, "responses_expected_classes_fingerprint_word.csv"))
    for i, (e, r) in enumerate(zip(expectedClasses, resultClasses)):
      if sorted(e) != sorted(r): print i, e, r
    [self.assertEqual(sorted(e), sorted(r),
      "Fingerprint model predicted classes other than what we expect.")
      for e, r in zip(expectedClasses, resultClasses)]


  def testClassifyEndpointAsExpected(self):
    """
    Tests ClassificationModelEndpoint.
    
    Training on the first five samples of the dataset, and testing on the rest,
    the model's classifications should match those in the expected classes
    data file.
    """
    modelName = "CioEndpoint"
    runner = Runner(dataPath=os.path.join(DATA_DIR, "responses.csv"),
                    resultsDir="",
                    experimentName="endpoint_test",
                    loadPath=None,
                    modelName=modelName,
                    numClasses=3,
                    plots=0,
                    orderedSplit=True,
                    trainSizes=[5],
                    verbosity=0)
    runner.initModel(modelName)
    self.runExperiment(runner)

    expectedClasses, resultClasses = self.getExpectedClassifications(runner,
      os.path.join(DATA_DIR, "responses_expected_classes_endpoint.csv"))

    [self.assertEqual(sorted(e), sorted(r),
      "Endpoint model predicted classes other than what we expect.")
      for e, r in zip(expectedClasses, resultClasses)]


  def testClassifyHTMAsExpectedWithKNN(self):
    """
    Tests ClassificationModelHTM.
    
    Training on the first five samples of the dataset, and testing on the rest,
    the model's classifications should match those in the expected classes
    data file.
    """
    modelName = "HTMNetwork"
    runner = HTMRunner(dataPath=os.path.join(DATA_DIR, "responses_network.csv"),
                       networkConfigPath=os.path.join(
                         DATA_DIR, "network_config_sp_tm_knn.json"),
                       resultsDir="",
                       experimentName="htm_test",
                       loadPath=None,
                       modelName=modelName,
                       numClasses=3,
                       plots=0,
                       orderedSplit=True,
                       trainSizes=[5],
                       verbosity=0,
                       generateData=False,
                       votingMethod="last",
                       classificationFile=os.path.join(
                         DATA_DIR, "responses_classifications.json"))
    runner.initModel(0)
    runner.runExperiment()

    expectedClasses, resultClasses = self.getExpectedClassifications(runner,
      os.path.join(DATA_DIR, "responses_expected_classes_htm.csv"))

    [self.assertEqual(sorted(e), sorted(r),
      "HTM model predicted classes other than what we expect.")
      for e, r in zip(expectedClasses, resultClasses)]


# TODO: add the following tests...

#  def testClassifyHTMAsExpectedWithCLA(self):

#  def testTrainOnAllTestOnAll(self):
#    """Train on all samples, save model, load model, and test on all samples."""


if __name__ == "__main__":
     unittest.main()
