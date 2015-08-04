#!/usr/bin/env python
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
This file contains a class that tokenizes, randomizes, and writes the data to a
file in the format of the network API.
"""

import argparse
import csv
import os
import pprint
import random
import string

from collections import defaultdict
from fluent.utils.csv_helper import readCSV
from fluent.utils.text_preprocess import TextPreprocess

try:
  import simplejson as json
except ImportError:
  import json



class NetworkDataGenerator(object):
  """Class for generating data in the format for a record stream."""


  def  __init__(self):
    """
    Column headers are marked "private" with a leading underscore in order to
    distingush them from dictinonary keys used in the Network API.

    Note: a reset marks the first item of a new sequence.
    """
    self.records = []
    self.fieldNames = ["_token", "_categories", "_sequenceID", "_reset"]
    self.types = {"_token": "string",
                  "_categories": "list",
                  "_sequenceID": "int",
                  "_reset": "int"}
    self.specials = {"_token": "",
                     "_categories": "C",
                     "_sequenceID": "S",
                     "_reset": "R"}

    # len(self.categoryToId) gives each category a unique id w/o having
    # duplicates
    self.categoryToId = defaultdict(lambda: len(self.categoryToId))


  def split(self, filename, sampleIdx, numLabels, textPreprocess, abbrCSV="",
      contrCSV="", ignoreCommon=None, removeStrings=None, correctSpell=False,
      **kwargs):
    """
    Split all the comments in a file into tokens. Preprocess if necessary.
    @param filename        (str)    Path to csv file
    @param sampleIdx       (int)    Column number of the text sample
    @param numLabels       (int)    Number of columns of category labels.
    @param textPreprocess  (bool)   True will preprocess text while tokenizing.
    Please see TextPreprocess tokenize() for the other parameters
    """
    dataDict = readCSV(filename, sampleIdx, numLabels)
    if dataDict is None:
      raise Exception("Could not read CSV.")

    preprocessor = TextPreprocess(abbrCSV=abbrCSV, contrCSV=contrCSV)
    expandAbbr = (abbrCSV != "")
    expandContr = (contrCSV != "")

    for i, idx in enumerate(dataDict.keys()):
      comment, categories = dataDict[idx]
      # Convert the categories to a string of their IDs
      categories = string.join([str(self.categoryToId[c]) for c in categories])

      if textPreprocess:
        tokens = preprocessor.tokenize(comment, ignoreCommon, removeStrings,
            correctSpell, expandAbbr, expandContr)
      else:
        tokens = preprocessor.tokenize(comment)

      # Write the sequence of data records for this sample.
      record = {"_categories":categories,
                "_sequenceID":i}
      data = []
      reset = 1
      for t in tokens:
        tokenRecord = record.copy()
        tokenRecord["_token"] = t
        tokenRecord["_reset"] = reset
        reset = 0
        data.append(tokenRecord)

      self.records.append(data)


  def randomizeData(self):
    random.shuffle(self.records)


  def saveData(self, dataOutputFile, categoriesOutputFile, **kwargs):
    """
    Save the processed data and the associated category mapping.
    @param dataOutputFile       (str)   Location to save data
    @param categoriesOutputFile (str)   Location to save category map
    @return                     (str)   Path to the saved data file iff
                                        saveData() is successful.
    """
    if self.records is None:
      return False

    if not dataOutputFile.endswith("csv"):
      raise TypeError("data output file must be csv.")
    if not categoriesOutputFile.endswith("json"):
      raise TypeError("category output file must be json")

    # Ensure directory exists
    dataOutputDirectory = os.path.dirname(dataOutputFile)
    if not os.path.exists(dataOutputDirectory):
      os.makedirs(dataOutputDirectory)

    categoriesOutputDirectory = os.path.dirname(categoriesOutputFile)
    if not os.path.exists(categoriesOutputDirectory):
      os.makedirs(categoriesOutputDirectory)

    with open(dataOutputFile, 'w') as f:
      # Header
      writer = csv.DictWriter(f, fieldnames=self.fieldNames)
      writer.writeheader()

      # Types
      writer.writerow(self.types)

      # Special characters
      writer.writerow(self.specials)

      for data in self.records:
        for record in data:
          writer.writerow(record)

    with open(categoriesOutputFile, 'w') as f:
      f.write(json.dumps(self.categoryToId,
                         sort_keys=True,
                         indent=4,
                         separators=(',', ': ')))

    return dataOutputFile


  def reset(self):
    self.records = []
    self.fieldNames = ["token", "_sequenceID", "_reset"]
    self.types = {"token": "string",
                  "_sequenceID": "int",
                  "_reset": "int"}
    self.specials = {"token": "",
                     "_sequenceID": "S",
                     "_reset": "R"}

    self.categoryToId.clear()


  @staticmethod
  def getResetsIndices(networkDataFile):
    """Returns the indices at which the data sequences reset."""
    try:
      with open(networkDataFile) as f:
        reader = csv.reader(f)
        next(reader, None)
        next(reader, None)
        resetIdx = next(reader).index("R")

        resets = []
        for i, line in enumerate(reader):
          print line
          if int(line[resetIdx]) == 1:
            resets.append(i)
        return resets

    except IOError as e:
      print "Could not open the file {}.".format(networkDataFile)
      raise e


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Create data for network API")

  parser.add_argument("-fin", "--filename",
                      type=str,
                      required=True,
                      help="path to input file. REQUIRED")
  parser.add_argument("-fo", "--dataOutputFile",
                      default="network_experiment/data.csv",
                      type=str,
                      help="File to write processed data to.")
  parser.add_argument("-c", "--categoriesOutputFile",
                      type=str,
                      default="network_experiment/categories.json",
                      help="File to write the categories to ID mapping.")
  parser.add_argument("--sampleIdx",
                      type=int,
                      default=2,
                      help="Column number of the text sample.")
  parser.add_argument("--numLabels",
                      type=int,
                      default=3,
                      help="Column number(s) of the category label.")
  parser.add_argument("--textPreprocess",
                      action="store_true",
                      default=False,
                      help="Basic preprocessing. Use specific tags for custom "
                      "preprocessing")
  parser.add_argument("--ignoreCommon",
                      default=100,
                      type=int,
                      help="Number of common words to ignore")
  parser.add_argument("--removeStrings",
                      type=str,
                      default=["[identifier deleted]"],
                      nargs="+",
                      help="Strings to remove in preprocessing")
  parser.add_argument("--correctSpell",
                      default=True,
                      action="store_false",
                      help="Whether or not to use spelling correction")
  parser.add_argument("--abbrCSV",
                      default="",
                      help="Path to CSV of abbreviations data")
  parser.add_argument("--contrCSV",
                      default="",
                      help="Path to CSV of contractions data")
  parser.add_argument("--randomize",
                      default=False,
                      action="store_true",
                      help="Whether or not to randomize the order of the data "
                      "samples before saving")

  options = vars(parser.parse_args())

  pprint.pprint(options)
  print ("Note: preprocessing params only take affect if textPreprocess "
         "argument is set.")

  dataGenerator = NetworkDataGenerator()
  dataGenerator.split(**options)

  if options["randomize"]:
    dataGenerator.randomizeData()

  outFile = dataGenerator.saveData(**options)
