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
"""
Initial experiment runner for classification survey question responses.

EXAMPLE: from the fluent directory, run...
  python experiments/baseline_experiment.py
  data/sample_reviews/sample_reviews_data_training.csv

  - The runner sets up the data path such that the experiment runs on a single
  data file located in the nupic.fluent/data directory. The data path MUST BE
  SPECIFIED at the cmd line.
  - This example runs the ClassificationModelRandomSDR subclass of Model. To use
  a different model, use cmd line args modelName and modelModuleName.
  - The call to readCSV() below is specific for the format of this data file,
  and should be changed for CSVs with different columns.

Please note the following definitions:
- k-fold cross validation: the training dataset is split
  differently for each of the k trials. The majority of the dataset is used for
  training, and a small portion (1/k) is held out for evaluation; this
  evaluation data is different from the test data.
- classification and label are used interchangeably
"""


import argparse
import cPickle as pkl
import itertools
import numpy
import os
import time
from collections import defaultdict

from fluent.utils.csv_helper import readCSV, writeFromDict
from fluent.utils.data_split import KFolds
from fluent.utils.text_preprocess import TextPreprocess


def runExperiment(model, patterns, idxSplits, batch):
  """
  Trains the model on patterns specified by the first entry of idxSplits, then
  tests on the patterns of the second entry on idxSplits.

  @param model          (Model)       Classification model instance.
  @param patterns       (list)        Each item is a dict with the sample
                                      encoding a numpy array bitmap in field
                                      "bitmap".
  @param idxSplits      (tuple)       Tuple of train/eval split data indices.
  @param batch          (bool)        Whether or not to train on all the data
                                      in a batch
  @return                             Return same as testing().
  """
  model.resetModel()
  training(model, [patterns[i] for i in idxSplits[0]], batch)
  return testing(model, [patterns[i] for i in idxSplits[1]])


def training(model, trainSet, batch):
  """
  Trains model on the bitmap patterns and corresponding labels lists one at a
  time (i.e. streaming).
  @param batch          (bool)        Whether or not to train on all the data
                                      in a batch
  """
  if batch:
    samples = [s["pattern"] for s in trainSet]
    labels = [s["labels"] for s in trainSet]
    model.trainModel(samples, labels)
  else:
    for sample in trainSet:
      model.trainModel([sample["pattern"]], [sample["labels"]])


def testing(model, evalSet):
  """
  Tests model on the bitmap patterns and corresponding labels lists, one at a
  time (i.e. streaming).

  @return trialResults    (list)      List of two lists, where the first list
      is the model's predicted classifications, and the second list is the
      actual classifications.
  """
  trialResults = ([], [])
  for sample in evalSet:
    predicted = model.testModel(sample["pattern"])
    trialResults[0].append(predicted)
    trialResults[1].append(sample["labels"])
  return trialResults


def calculateResults(model, results, refs, indices, fileName):
  """
  Evaluate the results, returning accuracy and confusion matrix, and writing
  the confusion matrix to a CSV.

  TODO: csv writing broken until ClassificationModel confusion matrix is fixed
  """
  result = model.evaluateResults(results, refs, indices)
  # result[1].to_csv(fileName)
  return result


def computeExpectedAccuracy(predictedLabels, dataPath):
  """
  Compute the accuracy of the models predictions against what we expect it to
  predict; considers only single classification.
  """
  dataDict = readCSV(dataPath, 2, [3])
  expectedLabels = [data[1] for _, data in dataDict.iteritems()]

  if len(expectedLabels) != len(predictedLabels):
    raise ValueError("Lists of labels must have the same length.")

  accuracy = len([i for i in xrange(len(expectedLabels))
    if expectedLabels[i]==predictedLabels[i]]) / float(len(expectedLabels))

  print "Accuracy against expected classifications = ", accuracy


def setupData(args):
  """ Performs data preprocessing and setup given the user-specified args.

  @param args       (Namespace)     User-provided arguments via the cmd line.
  @return           (tuple)         Tuple where first entry is a list of the
      samples, the second is the list of gold labels per example, the third is
      the list of all possible labels, and the fourth is the labels per example
      in the data.
  """
  dataDict = readCSV(args.dataPath, 2, args.numLabels)

  # Collect each possible label string into a list, where the indices will be
  # their references throughout the experiment.
  labelReference = list(set(
    itertools.chain.from_iterable(map(lambda x: x[1], dataDict.values()))))

  for idx, data in dataDict.iteritems():
    comment, labels = data
    dataDict[idx] = (comment, numpy.array([labelReference.index(label)
                                    for label in labels],
                                    dtype="int8"))

  texter = TextPreprocess(abbrCSV=args.abbrCSV, contrCSV=args.contrCSV)
  expandAbbr = (args.abbrCSV != "")
  expandContr = (args.contrCSV != "")
  if args.textPreprocess:
    samples = [(texter.tokenize(data[0],
                                ignoreCommon=100,
                                removeStrings=["[identifier deleted]"],
                                correctSpell=True,
                                expandAbbr=expandAbbr,
                                expandContr=expandContr),
               data[1]) for _, data in dataDict.iteritems()]
  else:
    samples = [(texter.tokenize(data[0]), data[1])
               for _, data in dataDict.iteritems()]

  return samples, labelReference


def run(args):
  """
  The experiment is configured to run on question response data.

  To run k-folds cross validation, arguments must be: kFolds > 1, train = False,
  test = False. To run either training or testing, kFolds = 1.
  """
  start = time.time()

  # Setup directories.
  root = os.path.dirname(__file__)
  modelPath = os.path.abspath(
    os.path.join(root, args.resultsDir, args.expName, args.modelName))
  if not os.path.exists(modelPath):
    os.makedirs(modelPath)

  # Verify input params.
  if not os.path.isfile(args.dataPath):
    raise ValueError("Invalid data path.")
  if (not isinstance(args.kFolds, int)) or (args.kFolds < 1):
    raise ValueError("Invalid value for number of cross-validation folds.")
  if (args.train or args.test) and args.kFolds > 1:
    raise ValueError("Experiment runs either k-folds CV or training/testing, "
                     "not both.")

  # Load or init model.
  if args.load:
    with open(
      os.path.join(modelPath, "model.pkl"), "rb") as f:
      model = pkl.load(f)
    print "Model loaded from \'{0}\'.".format(modelPath)
  else:
    try:
      module = __import__(args.modelModuleName, {}, {}, args.modelName)
      modelClass = getattr(module, args.modelName)
      model = modelClass(verbosity=args.verbosity,
                         numLabels=args.numLabels)
    except ImportError:
      raise RuntimeError("Could not find model class \'%s\' to import."
                         % args.modelName)

  print "Reading in data and preprocessing."
  preprocessTime = time.time()

  samples, labelReference = setupData(args)

  print("Preprocessing complete; elapsed time is {0:.2f} seconds.".
        format(time.time() - preprocessTime))
  if args.verbosity > 1:
    for i, s in enumerate(samples): print i, s, labelReference[labels[i]]

  print "Encoding the data."
  encodeTime = time.time()
  patterns = [{"pattern": model.encodePattern(s[0]),
              "labels": s[1]}
              for s in samples]

  print("Done encoding; elapsed time is {0:.2f} seconds.".
        format(time.time() - encodeTime))
  model.writeOutEncodings(patterns, modelPath)

  # Either we train on all the data, test on all the data, or run k-fold CV.
  if args.train:
    training(model, patterns, args.batch)

  if args.test:
    results = testing(model, patterns)
    resultMetrics = calculateResults(
      model, results, labelReference, xrange(len(samples)),
      os.path.join(modelPath, "test_results.csv"))
    print resultMetrics
    if model.plot:
      model.plotConfusionMatrix(resultMetrics[1])

  elif args.kFolds > 1:
    # Run k-folds cross validation -- train the model on a subset, and evaluate
    # on the remaining subset.
    partitions = KFolds(args.kFolds).split(range(len(samples)), randomize=True)
    intermResults = []
    predictions = []
    resultsDict = defaultdict(list)
    for k in xrange(args.kFolds):
      print "Training and testing for CV fold {0}.".format(k)
      kTime = time.time()
      trialResults = runExperiment(model, patterns, partitions[k], args.batch)
      print("Fold complete; elapsed time is {0:.2f} seconds.".format(
            time.time() - kTime))
      
      # Populate resultsDict for writing out the classifications.
      for i, sampleNum in enumerate(partitions[k][1]):
        sample = samples[sampleNum][0]
        pred = sorted([labelReference[j] for j in trialResults[0][i]])
        actual = sorted([labelReference[j] for j in trialResults[1][i]])
        resultsDict[sampleNum] = (sample, actual, pred)
      
      if args.expectationDataPath:
        # Keep the predicted labels (top prediction only) for later.
        p = [l if l else [None] for l in trialResults[0]]
        predictions.append(
          [labelReference[idx[0]] if idx[0] != None else '(none)' for idx in p])

      print "Calculating intermediate results for this fold. Writing to CSV."
      (accuracy, cm) = calculateResults(
        model, trialResults, labelReference, partitions[k][1],
        os.path.join(modelPath, "evaluation_fold_" + str(k) + ".csv"))
      print "Accuracy after fold %d is %f" %(k, accuracy)

      intermResults.append((accuracy, cm))

    print "Calculating cumulative results for {0} trials.".format(args.kFolds)
    results = model.evaluateCumulativeResults(intermResults)

    # TODO: csv writing broken until ClassificationModel confusion matrix is fixed
    # results["total_cm"].to_csv(os.path.join(modelPath, "evaluation_totals.csv"))
    if args.expectationDataPath:
      computeExpectedAccuracy(list(itertools.chain.from_iterable(predictions)),
        os.path.abspath(os.path.join(root, '../..', args.expectationDataPath)))

  ## TODO:
  # print "Calculating random classifier results for comparison."
  # print model.classifyRandomly(labels)

  print "Saving model to \'{0}\' directory.".format(modelPath)
  with open(
    os.path.join(modelPath, "model.pkl"), "wb") as f:
    pkl.dump(model, f)
  print "Experiment complete in {0:.2f} seconds.".format(time.time() - start)

  resultsPath = os.path.join(modelPath, "results.csv")
  print "Saving results to {}.".format(resultsPath)
  headers = ("Tokenized sample", "Actual", "Predicted")
  writeFromDict(resultsDict, headers, resultsPath)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("dataPath",
                      help="Absolute path to data CSV.")
  parser.add_argument("--expectationDataPath",
                      default="",
                      type=str,
                      help="Path from fluent root directory to the file with "
                      " expected labels.")
  parser.add_argument("-k", "--kFolds",
                      default=5,
                      type=int,
                      help="Number of folds for cross validation; k=1 will "
                      "run no cross-validation.")
  parser.add_argument("--expName",
                      default="survey_response_sample",
                      type=str,
                      help="Experiment name.")
  parser.add_argument("-m", "--modelName",
                      default="ClassificationModelKeywords",
                      type=str,
                      help="Name of model class. Also used for model results "
                      "directory and pickle checkpoint.")
  parser.add_argument("-mm", "--modelModuleName",
                      default="fluent.models.classify_keywords",
                      type=str,
                      help="model module (location of model class).")
  parser.add_argument("--numLabels",
                      help="Specifies the number of classes per sample.",
                      type=int,
                      default=3)
  parser.add_argument("--textPreprocess",
                      type=bool,
                      help="Whether to preprocess text",
                      default=False)
  parser.add_argument("--contrCSV",
                      default="",
                      help="Path to contraction csv")
  parser.add_argument("--abbrCSV",
                      default="",
                      help="Path to abbreviation csv")
  parser.add_argument("--batch",
                      help="Train the model with all the data at one time",
                      action="store_true")
  parser.add_argument("--load",
                      help="Load the serialized model.",
                      default=False)
  parser.add_argument("--train",
                      help="Train the model on all the data.",
                      default=False)
  parser.add_argument("--test",
                      help="Test the model on all the data.",
                      default=False)
  parser.add_argument("--resultsDir",
                      default="results",
                      help="This will hold the evaluation results.")
  parser.add_argument("--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                      "verbosity 1 will include results, and verbosity > 1 "
                      "will print out preprocessed tokens and kNN inference "
                      "metrics.")
  args = parser.parse_args()
  run(args)
