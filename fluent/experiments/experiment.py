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

TODO: after merging PlotNLP code, set the defualt plotting to 1.
"""


import argparse
import os
import pprint
import time

from fluent.experiments.runner import Runner
from fluent.experiments.multi_runner import MultiRunner
from fluent.utils.plotting import PlotNLP


def checkInputs(args):
  """Function that displays a set of arguments and asks to proceed."""
  pprint.pprint(vars(args))
  userIn = raw_input("Proceed? (y/n): ")

  if userIn == 'y':
    return True

  if userIn == 'n':
    return False

  print "Incorrect input given\n"
  return checkInputs(args)


def run(args):
  start = time.time()

  root = os.path.dirname(os.path.realpath(__file__))
  resultsDir = os.path.join(root, args.resultsDir)

  if os.path.isdir(args.dataPath):
    runner = MultiRunner(dataPath=args.dataPath,
                         resultsDir=resultsDir,
                         experimentName=args.experimentName,
                         load=args.load,
                         modelName=args.modelName,
                         modelModuleName=args.modelModuleName,
                         numClasses=args.numClasses,
                         plots=args.plots,
                         orderedSplit=args.orderedSplit,
                         trainSize=args.trainSize,
                         verbosity=args.verbosity,
                         test=args.test)
  else:
    runner = Runner(dataPath=args.dataPath,
                    resultsDir=resultsDir,
                    experimentName=args.experimentName,
                    load=args.load,
                    modelName=args.modelName,
                    modelModuleName=args.modelModuleName,
                    numClasses=args.numClasses,
                    plots=args.plots,
                    orderedSplit=args.orderedSplit,
                    trainSize=args.trainSize,
                    verbosity=args.verbosity)

  runner.initModel()

  print "Reading in data and preprocessing."
  dataTime = time.time()
  runner.setupData()
  print ("Data setup complete; elapsed time is {0:.2f} seconds.\nNow encoding "
        "the data".format(time.time() - dataTime))

  encodeTime = time.time()
  runner.encodeSamples()
  print ("Encoding complete; elapsed time is {0:.2f} seconds.\nNow running the "
         "experiment.".format(time.time() - encodeTime))

  runner.runExperiment()

  runner.calculateResults()

  runner.save()

  print "Experiment complete in {0:.2f} seconds.".format(time.time() - start)

  if args.validation:
    print "Validating experiment against expected classifications..."
    print runner.validateExperiment(args.validation)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("dataPath",
                      help="absolute path to data CSV or folder of CSVs.")
  parser.add_argument("--test", default=None,
                      help="path to data CSV to use for testing if provided.\
                      Otherwise will test on \'dataPath\'")
  parser.add_argument("-e", "--experimentName",
                      default="survey_baseline_example",
                      type=str,
                      help="Experiment name.")
  parser.add_argument("-m", "--modelName",
                      default="ClassificationModelRandomSDR",
                      type=str,
                      help="Name of model class. Also used for model results "
                           "directory and pickle checkpoint.")
  parser.add_argument("-mm", "--modelModuleName",
                      default="fluent.models.classify_random_sdr",
                      type=str,
                      help="Model module (location of model class).")
  parser.add_argument("--resultsDir",
                      default="results",
                      help="This will hold the experiment results.")
  parser.add_argument("--load",
                      help="Load the serialized model.",
                      default=False)
  parser.add_argument("--numClasses",
                      help="Specifies the number of classes per sample.",
                      type=int,
                      default=3)
  parser.add_argument("--plots",
                      default=1,
                      type=int,
                      help="0 for no evaluation plots, 1 for classification "
                           "accuracy plots, 2 includes the confusion matrix.")
  parser.add_argument("--orderedSplit",
                      default=False,
                      action="store_true",
                      help="To split the train and test sets, False will split "
                            "the samples randomly, True will allocate the "
                            "first n samples to training with the remainder "
                            "for testing.")
  parser.add_argument("--trainSize",
                      default=[7, 7, 7, 13, 13, 13],
                      nargs="+",
                      type=int,
                      help="Number of samples to use in training. Separate "
                           "with spaces for multiple trials.")
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                           "verbosity 1 will include results, and verbosity > "
                            "1 will print out preprocessed tokens and kNN "
                            "inference metrics.")
  parser.add_argument("--validation",
                      default="",
                      help="Path to file of expected classifications.")
  parser.add_argument("--skipConfirmation",
                      help="If specified will skip the user confirmation step",
                      default=False,
                      action="store_true")

  args = parser.parse_args()

  if args.skipConfirmation or checkInputs(args):
    run(args)
