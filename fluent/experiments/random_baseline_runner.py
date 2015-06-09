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
"""
Experiment runner for classification survey question responses.
Please note the following definitions:
- Training dataset: all the data files used for experimentally building the NLP
  system. During k-fold cross validation, the training dataset is split
  differently for each of the k trials. The majority of the dataset is used for
  training, and a small portion (1/k) is held out for evaluation; this
  evaluation data is different from the test data.
- Testing dataset: the data files held out until the NLP system is complete.
  That is, the system should never see this testing data and then go back and
  change models/params/methods/etc. at the risk of overfitting.
- Classification and label are used interchangeably.

Each sample is a token of text, for which there are multiple within a single
question response. The samples of a single response all correspond to the
classification(s) for the response. There can be one or more classifications per
sample, which are in separate columns of the input CSV.

The model learns each sample (token) independently, encoding each w/ a
random SDR, which is fed into a kNN classifier. For a given response in an
evaluation (and test) dataset, each token is independently classified, and the
response is then labeled with the top classification(s) amongst its tokens.
"""

import argparse
import cPickle as pkl
import itertools
import numpy
import os
import time

from fluent.utils.csv_helper import readCSV
from fluent.utils.text_preprocess import TextPreprocess
from fluent.models.classify_randomSDR import ClassificationModelRandomSDR


def training(model, trainSet):
  """Trains model on the bitmap patterns and corresponding labels lists."""
  for x in trainSet:
    model.trainModel(x[0], x[1])


def testing(model, evalSet):
  """
  Tests model on the bitmap patterns and corresponding labels lists.

  @return trialResults    (list)            List of two lists, where the first
                                            list is the model's predicted
                                            classifications, and the second list
                                            is the actual classifications.
  """
  trialResults = [[], []]
  for x in evalSet:
    predicted = model.testModel(x[0])
    trialResults[0].append(predicted)
    trialResults[1].append(x[1])
  return trialResults


def computeExpectedAccuracy(predictedLabels, dataPath):
  """
  Compute the accuracy of the models predictions against what we expect it to
  predict; considers only single classification.
  """
  _, expectedLabels = readCSV(dataPath, 2, [3])
  if len(expectedLabels) != len(predictedLabels):
    raise ValueError("Lists of labels must have the same length.")

  accuracy = len([i for i in xrange(len(expectedLabels))
    if expectedLabels[i]==predictedLabels[i]]) / float(len(expectedLabels))

  print "Accuracy against expected classifications = ", accuracy


def run(args):
  """
  The experiment is configured to run on question response data.

  The runner sets up the data path to such that the experiment runs on a single
  data file located in the nupic.fluent/data directory.
  The data path MUST BE SPECIFIED at the cmd line, e.g. from the fluent dir:

  python experiments/random_baseline_runner.py data/sample_reviews/sample_reviews_data_training.csv

  To run k-folds cross validation, arguments must be: kFolds > 1, train = False,
  test = False. To run either training or testing, kFolds = 1.
  """
  start = time.time()

  # Setup directories.
  root = os.path.dirname(__file__)
  dataPath = os.path.abspath(os.path.join(root, '../..', args.dataFile))
  modelPath = os.path.abspath(
    os.path.join(root, args.resultsDir, args.expName, args.modelName))
  if not os.path.exists(modelPath):
    os.makedirs(modelPath)

  # Verify input params.
  if not os.path.isfile(dataPath):
    raise ValueError("Invalid data path.")
  if (not isinstance(args.kFolds, int)) or (args.kFolds < 1):
    raise ValueError("Invalid value for number of cross-validation folds.")
  if args.train and args.test:
    raise ValueError("Run training and testing independently.")
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
    model = ClassificationModelRandomSDR(verbosity=args.verbosity)

  # Get and prep data.
  texter = TextPreprocess()
  samples, labels = readCSV(dataPath, 2, [3])  # Y data, [3] -> range(3,6)
  labelReference = list(set(labels))
  labels = numpy.array([labelReference.index(l) for l in labels], dtype=int)
  split = len(samples)/args.kFolds
  samples = [texter.tokenize(sample,
                             ignoreCommon=100,
                             removeStrings=["[identifier deleted]"],
                             correctSpell=True)
             for sample in samples]
  if args.verbosity > 1:
    for i, s in enumerate(samples): print i, s, labelReference[labels[i]]
  patterns = [[model.encodePattern(t) for t in tokens] for tokens in samples]

  # Either we train on all the data, test on all the data, or run k-fold CV.
  if args.train:
    training(model,
      [(p, labels[i]) for i, p in enumerate(patterns)])
  elif args.test:
    trialResults = testing(model,
      [(p, labels[i]) for i, p in enumerate(patterns)])
  elif args.kFolds>1:
    intermResults = []
    predictions = []
    for k in range(args.kFolds):
      # Train the model on a subset, and hold the evaluation subset.
      model.resetModel()
      evalIndices = range(k*split, (k+1)*split)
      trainIndices = [i for i in range(len(samples)) if not i in evalIndices]

      print "Training for CV fold {0}.".format(k)
      training(model,
        [(patterns[i], labels[i]) for i in trainIndices])

      print "Evaluating for trial {0}.".format(k)
      trialResults = testing(model,
        [(patterns[i], labels[i]) for i in evalIndices])

      if args.expectationDataPath:
        # Keep the predicted labels (top prediction only) for later.
        p = [l if l else [None] for l in trialResults[0]]
        predictions.append([labelReference[idx[0]] if idx[0] != None else '(none)' for idx in p])

      print "Calculating intermediate results for this fold."
      result = model.evaluateTrialResults(
        trialResults, labelReference, evalIndices)
      intermResults.append(result)
      result[1].to_csv(os.path.join(
        modelPath, "evaluation_fold_" + str(k) + ".csv"))

    print "Calculating cumulative results for {0} trials.".format(args.kFolds)
    results = model.evaluateFinalResults(intermResults)
    results["total_cm"].to_csv(os.path.join(modelPath, "evaluation_totals.csv"))
    if args.expectationDataPath:
      computeExpectedAccuracy(list(itertools.chain.from_iterable(predictions)),
        os.path.abspath(os.path.join(root, '../..', args.expectationDataPath)))

    print "Calculating random classifier results for comparison."
    # model.classifyRandomly(samples, labels)

  print "Saving model to \'{0}\' directory.".format(modelPath)
  with open(
    os.path.join(modelPath, "model.pkl"), "wb") as f:
    pkl.dump(model, f)
  print "Experiment complete in {0:.2f} seconds.".format(time.time() - start)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("dataFile")
  parser.add_argument("--expectationDataPath",
                      default="",
                      type=str,
                      help="Path from fluent root directory to the file with "
                      " expected labels.")
  parser.add_argument("-k", "--kFolds",
                      default=5,
                      type=int,
                      help="Number of folds for cross validation; k=1 will "
                      "train on "
                      "run no cross-validation.")
  parser.add_argument("--expName",
                      default="survey_response_random_sdr_training",
                      type=str,
                      help="Experiment name.")
  parser.add_argument("--modelName",
                      default="",
                      type=str,
                      help="Model name for pickle file.")
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
