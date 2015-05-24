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
classifications for the response; there can be one or more.

The model learns each sample (token) independently, encoding each w/ a 
random SDR, which is fed into a kNN classifier. For a given response in a
evaluation (and test) dataset, each token is independently classified, and the
response is then labeled with the top classification(s).
"""

import argparse
import csv
import os
import time

from fluent.bin.utils import readCSV, tokenize
from fluent.models.classify_randomSDR import ClassificationModelRandomSDR
from fluent.models.classification_model import ClassificationModel


def main(args):
  """
  The experiment is configured to run on question response data.

  The runner sets up the data path to such that the experiment runs on a single
  data file located in the nupic.fluent/data directory. The cmd line argument
  dataFile MUST BE SPECIFIED with the experiment folder and specific datafile, 
  e.g. 'sample_reviews/sample_reviews_data_q1.csv'.
  """
  start = time.time()

  # Setup directories:
  root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
  dataPath = os.path.abspath(os.path.join(
    root, '../..', args.dataFile))
  resultsPath = os.path.join(root, args.resultsDir, args.name)

  # Verify input params.
  if not os.path.isfile(dataPath):
    raise ValueError("Invalid data path.")
  if (not isinstance(args.kFolds, int)) or (args.kFolds < 1):
    raise ValueError("Invalid value for number of cross-validation folds.")

  # Load or init model:
  if args.load:
    model = ClassificationModel().load(resultsPath)
  else:
    model = ClassificationModelRandomSDR(kCV=args.kFolds,
                                         paths={"data":dataPath,
                                                "results":resultsPath},
                                         verbosity=args.verbosity)
  
  # Get and prep data:
  samples, labels = readCSV(dataPath)
  labelReference = list(set(labels))
  labels = [labelReference.index(l) for l in labels]
  split = len(samples)/args.kFolds
  samples = [tokenize(sample, ignoreCommon=True) for sample in samples]
  patterns = [[model.encodePattern(t) for t in tokens] for tokens in samples]

  # Run k-fold cross-validation:
  intermResults = []
  for k in range(args.kFolds):
    # Train the model on a subset, and hold the evaluation subset.
    evalIndices = range(k*split, (k+1)*split)
    trainIndices = [i for i in range(len(samples)) if not i in evalIndices]

    print "Training for CV fold %i." % k
    for i in trainIndices:
      model.trainModel(patterns[i], labels[i])

    print "Evaluating for trial %i." % k
    trialResults = [[], []]
    for i in evalIndices:
      predicted = model.testModel(patterns[i])
      if predicted == []:
        print "Skipping sample %i b/c no classification for this sample." % i
        continue
      trialResults[0].append(predicted)
      trialResults[1].append(labels[i])

    print "Calculating intermediate results for this fold."
    intermResults.append(
      model.evaluateTrialResults(trialResults, len(labelReference)))

  print "Calculating cumulative results for %i trials..." % k
  results = model.evaluateResults(intermResults)

  print "RESULTS..."
  print "max, mean, min accuracies = "
  print "%0.2f, %0.2f, %0.2f" % (results["max_accuracy"], results["mean_accuracy"], results["min_accuracy"])
  print "mean confusion matrix =\n", results["mean_cm"]

  print "Saving model to \'%s\' directory." % resultsPath
  model.save(resultsPath)
  print "Experiment complete in %0.2f seconds." % (time.time() - start)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("dataFile")
  parser.add_argument("-k", "--kFolds",
  										default=3,
  										type=int,
  										help="Number of folds for cross validation; k=1 will "
                      "run no cross-validation.")
  parser.add_argument("--name",
                      default="survey_response_random_sdr",
                      type=str,
                      help="Experiment name.")
  parser.add_argument("--load",
                      help="Load the checkpoint model.",
                      default=False)
  parser.add_argument("--train",
                      help="Train the model.",
                      default=True)
  parser.add_argument("--evaluate",
                      help="Run the model on the evaluation data.",
                      default=True)
  parser.add_argument("--test",  ## TODO: implement this?
                      help="Run the models on the test data.",
                      default=False)
  parser.add_argument("--resultsDir",
	                    default="results",
	                    help="This will hold the evaluation results.")
  parser.add_argument("--loadPath",
                      default=None,
                      help="Model checkpoint.")
  parser.add_argument("--verbosity",
                      help="Verbosity >0 will print out experiment progress.",
                      default=1)
  args = parser.parse_args()
  main(args)
