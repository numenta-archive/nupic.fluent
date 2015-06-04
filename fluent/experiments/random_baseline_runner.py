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
import cPickle as pkl
import numpy
import os
import time

from fluent.utils.read import readCSV
from fluent.utils.text_preprocess import TextPreprocess
from fluent.models.classify_randomSDR import ClassificationModelRandomSDR


def run(args):
  """
  The experiment is configured to run on question response data.

  The runner sets up the data path to such that the experiment runs on a single
  data file located in the nupic.fluent/data directory.
  The data path MUST BE SPECIFIED at the cmd line, e.g. from the fluent dir:

  python experiments/random_baseline_runner.py data/sample_reviews/sample_reviews_data_training.csv
  """
  start = time.time()

  # Setup directories.
  root = os.path.dirname(__file__)
  dataPath = os.path.abspath(os.path.join(root, '../..', args.dataFile))
  checkpointPklPath = os.path.abspath(
    os.path.join(root, args.resultsDir, args.name))

  # Verify input params.
  if not os.path.isfile(dataPath):
    raise ValueError("Invalid data path.")
  if (not isinstance(args.kFolds, int)) or (args.kFolds < 1):
    raise ValueError("Invalid value for number of cross-validation folds.")

  # Load or init model.
  if args.load:
    with open(os.path.join(checkpointPklPath, "model.pkl"), "rb") as f:
      model = pkl.load(f)
    print "Model loaded from \'{0}\'.".format(checkpointPklPath)
  else:
    model = ClassificationModelRandomSDR(verbosity=args.verbosity)

  # Get and prep data.
  texter = TextPreprocess()
  samples, labels = readCSV(dataPath)
  labelReference = list(set(labels))
  labels = numpy.array([labelReference.index(l) for l in labels], dtype=int)
  split = len(samples)/args.kFolds
  samples = [texter.tokenize(sample, 
                             ignoreCommon=100, 
                             removeStrings=["[identifier deleted]"],
                             correctSpell=True) 
             for sample in samples]
  patterns = [[model.encodePattern(t) for t in tokens] for tokens in samples]

  # Run k-fold cross-validation.
  intermResults = []
  for k in range(args.kFolds):
    # Train the model on a subset, and hold the evaluation subset.
    evalIndices = range(k*split, (k+1)*split)
    trainIndices = [i for i in range(len(samples)) if not i in evalIndices]

    if args.train:
      # First reset the model.
      model.resetModel()
      print "Training for CV fold {0}.".format(k)
      for i in trainIndices:
        model.trainModel(patterns[i], labels[i])

    if args.evaluate:
      print "Evaluating for trial {0}.".format(k)
      trialResults = [[], []]
      skippedIndices = []
      for i in evalIndices:
        predicted = model.testModel(patterns[i])
        if predicted == []:
          print("\tNote: skipping sample {0} b/c no classification for this "
            "sample.".format(i))
          skippedIndices.append(i)
          continue
        trialResults[0].append(predicted)
        trialResults[1].append(labels[i])

      # Evaluate this fold.
      print "Calculating intermediate results for this fold."
      [evalIndices.remove(idx) for idx in skippedIndices]
      intermResults.append(
        model.evaluateTrialResults(trialResults, labelReference, evalIndices))

  print "Calculating cumulative results for {0} trials.".format(k)
  results = model.evaluateFinalResults(intermResults)  ## TODO: model.writeResults to csv?

  print "Saving model to \'{0}\' directory.".format(checkpointPklPath)
  if not os.path.exists(checkpointPklPath):
    os.makedirs(checkpointPklPath)
  with open(os.path.join(checkpointPklPath, "model.pkl"), "wb") as f:
    pkl.dump(model, f)
  print "Experiment complete in {0:.2f} seconds.".format(time.time() - start)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("dataFile")
  parser.add_argument("-k", "--kFolds",
  										default=5,
  										type=int,
  										help="Number of folds for cross validation; k=1 will "
                      "run no cross-validation.")
  parser.add_argument("--name",
                      default="survey_response_random_sdr",
                      type=str,
                      help="Experiment name.")
  parser.add_argument("--load",
                      help="Load the serialized model.",
                      default=False)
  parser.add_argument("--train",
                      help="Train the model.",
                      default=True)
  parser.add_argument("--evaluate",
                      help="Run the model on the evaluation data.",
                      default=True)
  parser.add_argument("--resultsDir",
	                    default="results",
	                    help="This will hold the evaluation results.")
  parser.add_argument("--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment results, "
                      "verbosity 1 will print out training progress.")
  args = parser.parse_args()
  run(args)
