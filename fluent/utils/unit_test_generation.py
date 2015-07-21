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
This file contains CSV utility functions to use with nupic.fluent experiments.
"""

import argparse
import csv
import os
import random
import sys

from text_preprocess import TextPreprocess


def generateRandomFile(inputData, processor):
    """
    Generates a samples data file with all of the words per sample randomly
    shuffled.

    @param  (str)               Path to input original samples data file
    @param  (TextPreprocess)    Processor to perform some text cleanup.

    @return  NONE
    """
    with open(inputData, "r") as fRead:
      with open("sample_reviews_data_training_scramble.csv", "w") as fWrite:
        fileReader = csv.DictReader(fRead)
        fileWriter = csv.DictWriter(fWrite, fieldnames=fileReader.fieldnames)
        fileWriter.writeheader()
        for sample in fileReader:
          response = sample["Response"]
          tokens = processor.tokenize(response,
                                       removeStrings=["[identifier deleted]"])
          random.shuffle(tokens)
          shuffledResponse = " ".join(tokens)
          outputDict = {"QID": sample["QID"], "QuestionText":
              sample["QuestionText"], "Response": shuffledResponse,
              "Classification 1": sample["Classification 1"],
              "Classification 2": sample["Classification 2"],
              "Classification 3": sample["Classification 3"]}
          fileWriter.writerow(outputDict)


def generateReversedFile(inputData, processor):
    """
    Generates a samples data file with all of the words in the sample
    reversed.

    @param  (str)               Path to input original samples data file
    @param  (TextPreprocess)    Processor to perform some text cleanup.

    @return  NONE
    """
    with open(inputData, "r") as fRead:
      with open("sample_reviews_data_training_reverse.csv", "w") as fWrite:
        fileReader = csv.DictReader(fRead)
        fileWriter = csv.DictWriter(fWrite, fieldnames=fileReader.fieldnames)
        fileWriter.writeheader()
        for sample in fileReader:
          response = sample["Response"]
          tokens = processor.tokenize(response,
                                       removeStrings=["[identifier deleted]"])
          reversedResponse = " ".join(tokens[::-1])

          outputDict = {"QID": sample["QID"], "QuestionText":
              sample["QuestionText"], "Response": reversedResponse,
              "Classification 1": sample["Classification 1"],
              "Classification 2": sample["Classification 2"],
              "Classification 3": sample["Classification 3"]}
          fileWriter.writerow(outputDict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputData",
                        type=str,
                        help="Path to input data file")
    args = parser.parse_args()

    processor = TextPreprocess()
    generateRandomFile(args.inputData, processor)
    generateReversedFile(args.inputData, processor)
