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
file in the format of the network API
"""

import argparse
import csv
import json
import pandas
import random

from collections import defaultdict
from fluent.utils.text_preprocess import TextPreprocess


class NetworkDataGenerator(object):
  """Class for generating data for the network"""


  def  __init__(self):
    self.preprocessedData = []
    self.fieldNames = ["token", "_sequenceID", "_reset"]
    self.types = {"token": "str",
                  "_sequenceID": "int",
                  "_reset": "bool"}
    self.specials = {"token": "",
                     "_sequenceID": "S",
                     "_reset": "R"}

    # len(self.categoryToId) gives each category a unique id
    self.categoryToId = defaultdict(lambda: len(self.categoryToId))


  def preprocess(self, filename, sampleIdx, categoryIndices, abbrCSV="",
      contrCSV="", ignoreCommon=None, removeStrings=None, correctSpell=False,
      **kwargs):
    """
    Process all the comments in a file.
    @param filename        (str)    Path to csv file
    @param sampleIdx       (int)    Column number of the text sample
    @param categoryIndices (list)   List of numbers indicating the categories
    Please see TextPreprocess tokenize() for the other parameters
    """
    # Update header details
    for i in xrange(len(categoryIndices)):
      categoryKey = "_category{}".format(i)
      self.fieldNames.append(categoryKey)
      self.types[categoryKey] = "int"
      self.specials[categoryKey] = "C"
    
    dataTable = pandas.read_csv(filename).fillna('')
    numInstances, _ = dataTable.shape
    keys = dataTable.keys()
    categoryHeaders = [keys[i] for i in categoryIndices]

    textPreprocess = TextPreprocess(abbrCSV=abbrCSV, contrCSV=contrCSV)
    expandAbbr = (abbrCSV != "")
    expandContr = (contrCSV != "")

    for i in xrange(numInstances):
      # Get the category and convert it to an id
      categories = [self.categoryToId[dataTable[ch][i]] for ch in categoryHeaders]
      comment = dataTable[keys[sampleIdx]][i]

      tokens = textPreprocess.tokenize(comment, ignoreCommon, removeStrings,
        correctSpell, expandAbbr, expandContr)

      record = {"_category{}".format(i): c for i,c in enumerate(categories)}
      record["_sequenceID"] = i

      data = []
      reset = 1
      for t in tokens:
        tokenRecord = record.copy()
        tokenRecord["token"] = t
        tokenRecord["_reset"] = reset
        reset = 0
        data.append(tokenRecord)

      self.preprocessedData.append(data)


  def randomizeData(self):
    self.preprocesseData = random.shuffle(self.preprocessedData)


  def saveData(self, dataOutputFile, categoriesOutputFile, **kwargs):
    """
    Save the processed data and the associated category mapping
    @param dataOutputFile       (str)   Location to save data
    @param categoriesOutputFile (str)   Location to save category map
    """
    if self.preprocessedData is None:
      return False

    with open(dataOutputFile, 'w') as f:
      # Header
      writer = csv.DictWriter(f, fieldnames=self.fieldNames)
      writer.writeheader()

      # Types
      writer.writerow(self.types)

      # Special characters
      writer.writerow(self.specials)

      for data in self.preprocessedData:
        for record in data:
          writer.writerow(record)

    with open(categoriesOutputFile, 'w') as f:
      stringOfMap = json.dumps(self.categoryToId)
      f.write(stringOfMap)

    return True


  def reset(self):
    self.preprocessedData = []
    self.fieldNames = ["token", "_sequenceID", "_reset"]
    self.types = {"token": "str",
                  "_sequenceID": "int",
                  "_reset": "bool"}
    self.specials = {"token": "",
                     "_sequenceID": "S",
                     "_reset": "R"}

    self.categoryToId.clear()



def parse_args():
  parser = argparse.ArgumentParser(description="Create data file for network API")
  parser.add_argument("--filename", type=str, required=True,
    help="path to input file. REQUIRED")
  parser.add_argument("--sampleIdx", type=int, required=True,
    help="Column number of the text sample. REQUIRED")
  parser.add_argument("--categoryIndices", type=int, required=True, nargs="+",
    default=[], help="Column number(s) of the category label. REQUIRED")
  parser.add_argument("--dataOutputFile", default=None, type=str,
      required=True, help="File to write processed data to. REQUIRED")
  parser.add_argument("--categoriesOutputFile", default=None, type=str,
    required=True, help="File to write the categories to ID mapping. REQUIRED")

  parser.add_argument("--abbrCSV", default="", help="Path to abbreviation csv")
  parser.add_argument("--contrCSV", default="", help="Path to contraction csv")
  parser.add_argument("--ignoreCommon", default=None, type=int,
    help="Number of common words to ignore")
  parser.add_argument("--removeStrings", type=str, default=None, nargs="+",
    help="Strings to remove before tokenizing")
  parser.add_argument("--correctSpell", default=False, action="store_true",
    help="Whether or not to use spelling correction")

  parser.add_argument("--randomize", default=False, action="store_true",
    help="Whether or not to randomize the data before saving")

  return parser.parse_args()



if __name__ == "__main__":
  options = vars(parse_args())
  dataGenerator = NetworkDataGenerator()
  dataGenerator.preprocess(**options)

  if options["randomize"]:
    dataGenerator.randomizeData()

  dataGenerator.saveData(**options)
