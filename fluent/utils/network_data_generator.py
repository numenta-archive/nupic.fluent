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

import csv
import json
import pandas
import random

from fluent.utils.text_preprocess import TextPreprocess


class NetworkDataGenerator(object):
  """Class for generating data for the network"""


  def  __init__(self):
    self.preprocessedData = None
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
      contrCSV="", ignoreCommon=None, removeStrings=None, correctSpell=False):
    """
    Process all the comments in a file. Assumes the first column is the id
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
      sequenceID = dataTable[keys[0]][i]

      tokens = textPreprocess.tokenize(comment, ignoreCommon, removeStrings,
        correctSpell, expandAbbr, expandContr)

      record = {"_category{}".format(i): c for i,c in categories}
      record["_sequenceID"] = sequenceID

      data = []
      reset = 1
      for t in tokens:
        record["token"] = t
        record["_reset"] = reset
        reset = 0
        data.append(record)

      self.preprocessedData.append(data)


  def randomizeData(self):
    self.preprocesseData = random.shuffle(self.preprocessedData)


  def saveData(self, dataOutputFile, categoriesOutputFile):
    """
    Save the processed data and the associated category mapping
    """
    if self.preprocessedData is None:
      return False

    with open(dataOutputFile, 'w') as f:
      # Header
      writer = csv.DictWriter(f, fieldnames=self.fieldnames)
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
    self.preprocessedData = None
    self.fieldNames = ["token", "_sequenceID", "_reset"]
    self.types = {"token": "str",
                  "_sequenceID": "int",
                  "_reset": "bool"}
    self.specials = {"token": "",
                     "_sequenceID": "S",
                     "_reset": "R"}

    self.categoryToId.clear()
