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

from fluent.models.classify_randomSDR import ClassificationModelRandomSDR


def main(args):
  """
  The experiment is configured to run on question response data.

  The runner sets up the data path to such that the experiment runs on a single
  data file located in the nupic.fluent/data directory. The cmd line argument
  dataFile MUST BE SPECIFIED with the experiment folder and specific datafile, 
  e.g. 'sample_reviews/sample_reviews_data_q1.csv'.
  """
  # Setup directories.
  root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
  dataPath = os.path.abspath(os.path.join(
    root, '../..', args.dataFile))
  resultsPath = os.path.join(root, args.resultsDir, args.name)

  model = ClassificationModelRandomSDR(dataPath=dataPath,
                                       resultsPath=resultsPath,
                                       kCV=args.kFolds,
                                       train=args.train,
                                       evaluate=args.evaluate,
                                       test=args.test)
  
  results = model.runExperiment()
  print "RESULTS..."
  print "max, mean, min accuracies = "
  print "%0.2f, %0.2f, %0.2f" % (results["max_accuracy"], results["mean_accuracy"], results["min_accuracy"])
  print "mean confusion matrix =\n", results["mean_cm"]


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("dataFile")
  parser.add_argument("-k", "--kFolds",
  										default=3,
  										type=int,
  										help="Number of folds for cross validation, or None.")
  parser.add_argument("--name",
                      default="survey_response_random_sdr",
                      type=str,
                      help="Experiment name.")
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
  args = parser.parse_args()
  main(args)
