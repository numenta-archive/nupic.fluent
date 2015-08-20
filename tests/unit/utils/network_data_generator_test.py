#!/usr/bin/env python
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

"""Tests for the NetworkDataGenerator class."""

import os
import pandas
import random
import unittest

from fluent.utils.network_data_generator import NetworkDataGenerator
from nupic.data.file_record_stream import FileRecordStream

try:
  import simplejson as json
except:
  import json



class NetworkDataGeneratorTest(unittest.TestCase):


  def __init__(self, *args, **kwargs):
    super(NetworkDataGeneratorTest, self).__init__(*args, **kwargs)
    self.expected = [[{"_category0": "0",
                       "_category1": "1",
                       "_sequenceID": "0",
                       "token": "get",
                       "_reset": "1"},
                      {"_category0": "0",
                       "_category1": "1",
                       "_sequenceID": "0",
                       "token": "rid",
                       "_reset": "0"},
                      {"_category0": "0",
                       "_category1": "1",
                       "_sequenceID": "0",
                       "token": "of",
                       "_reset": "0"},
                      {"_category0": "0",
                       "_category1": "1",
                       "_sequenceID": "0",
                       "token": "the",
                       "_reset": "0"},
                      {"_category0": "0",
                       "_category1": "1",
                       "_sequenceID": "0",
                       "token": "trrible",
                       "_reset": "0"},
                      {"_category0": "0",
                       "_category1": "1",
                       "_sequenceID": "0",
                       "token": "kitchen",
                       "_reset": "0"},
                      {"_category0": "0",
                       "_category1": "1",
                       "_sequenceID": "0",
                       "token": "odor",
                       "_reset": "0"}],
                     [{"_category0": "2",
                       "_sequenceID": "1",
                       "token": "i",
                       "_reset": "1"},
                      {"_category0": "2",
                       "_sequenceID": "1",
                       "token": "don",
                       "_reset": "0"},
                      {"_category0": "2",
                       "_sequenceID": "1",
                       "token": "t",
                       "_reset": "0"},
                      {"_category0": "2",
                       "_sequenceID": "1",
                       "token": "care",
                       "_reset": "0"}]]
    self.dirName = os.path.dirname(os.path.realpath(__file__))


  def assertRecordsEqual(self, actual, expected):
    self.assertIsInstance(actual, list)
    self.assertEqual(len(actual), len(expected))
    for a, e in zip(actual, expected):
      self.assertEqual(len(a), len(e))
      for ra, re in zip(a, e):
        self.assertDictEqual(ra, re)


  def testSplitNoPreprocess(self):
    ndg = NetworkDataGenerator()
    filename = (self.dirName +
        "/../../../data/network_data_generator/multi_sample.csv")

    ndg.split(filename, 2, 3, False)
    self.assertRecordsEqual(ndg.records, self.expected)

  
  def testSplitPreprocess(self):
    ndg = NetworkDataGenerator()
    filename = (self.dirName +
        "/../../../data/network_data_generator/multi_sample.csv")

    expected = [[{"_category0": "0",
                  "_category1": "1",
                  "_sequenceID": "0",
                  "token": "get",
                  "_reset": "1"},
                 {"_category0": "0",
                  "_category1": "1",
                  "_sequenceID": "0",
                  "token": "rid",
                  "_reset": "0"},
                 {"_category0": "0",
                  "_category1": "1",
                  "_sequenceID": "0",
                  "token": "trouble",
                  "_reset": "0"},
                 {"_category0": "0",
                  "_category1": "1",
                  "_sequenceID": "0",
                  "token": "kitchen",
                  "_reset": "0"},
                 {"_category0": "0",
                  "_category1": "1",
                  "_sequenceID": "0",
                  "token": "odor",
                  "_reset": "0"}],
                [{"_category0": "2",
                  "_sequenceID": "1",
                  "token": "don",
                  "_reset": "1"},
                 {"_category0": "2",
                  "_sequenceID": "1",
                  "token": "care",
                  "_reset": "0"}]]

    ndg.split(filename, 2, 3, True, ignoreCommon=100, correctSpell=True)
    self.assertRecordsEqual(ndg.records, expected)


  def testRandomize(self):
    ndg = NetworkDataGenerator()
    filename = (self.dirName +
        "/../../../data/sample_reviews_multi/sample_reviews_data_training.csv")
    ndg.split(filename, 2, 3, False)

    random.seed(1)
    ndg.randomizeData()

    dataOutputFile = (self.dirName +
        "/../../../data/network_data_generator/multi_sample_split.csv")
    categoriesOutputFile = (self.dirName +
        "/../../../data/network_data_generator/multi_sample_categories.json")
    success = ndg.saveData(dataOutputFile, categoriesOutputFile)

    randomizedIDs = []
    dataTable = pandas.read_csv(dataOutputFile)
    for _, values in dataTable.iterrows():
      record = values.to_dict()
      idx = record["_sequenceID"]
      if idx.isdigit() and (not randomizedIDs or randomizedIDs[-1] != idx):
        randomizedIDs.append(idx)

    self.assertNotEqual(randomizedIDs, range(len(randomizedIDs)))


  def testSaveData(self):
    ndg = NetworkDataGenerator()
    filename = (self.dirName +
        "/../../../data/network_data_generator/multi_sample.csv")
    ndg.split(filename, 2, 3, False)
    dataOutputFile = (self.dirName +
        "/../../../data/network_data_generator/multi_sample_split.csv")
    categoriesOutputFile = (self.dirName +
        "/../../../data/network_data_generator/multi_sample_categories.json")
    success = ndg.saveData(dataOutputFile, categoriesOutputFile)
    self.assertTrue(success)

    dataTable = pandas.read_csv(dataOutputFile).fillna("")

    types = {"_category0": "int",
              "_category1": "int",
              "_category2": "int",
              "token": "string",
              "_sequenceID": "int",
              "_reset": "int"}
    specials = {"_category0": "C",
                "_category1": "C",
                "_category2": "C",
                "token": "",
                "_sequenceID": "S",
                "_reset": "R"}
    
    expected_records = [record for data in self.expected for record in data]
    expected_records.insert(0, specials)
    expected_records.insert(0, types)

    for idx, values in dataTable.iterrows():
      record = values.to_dict()
      if record["_category1"] == "":
        del record["_category1"]

      if record["_category2"] == "":
        del record["_category2"]

      self.assertDictEqual(record, expected_records[idx])

    with open(categoriesOutputFile) as f:
      categories = json.load(f)

    expected_categories = {"kitchen": 0, "environment": 1, "not helpful": 2}
    self.assertDictEqual(categories, expected_categories)


  def testSaveDataIncorrectType(self):
    ndg = NetworkDataGenerator()
    filename = (self.dirName +
        "/../../../data/network_data_generator/multi_sample.csv")
    dataOutputFile = (self.dirName +
        "/../../../data/network_data_generator/multi_sample_split.csv")
    categoriesOutputFile = (self.dirName +
        "/../../../data/network_data_generator/multi_sample_categories.csv")
    ndg.split(filename, 2, 3, False)

    with self.assertRaises(TypeError):
      ndg.saveData(dataOutputFile, categoriesOutputFile)


  def testFileRecordStreamReadData(self):
    ndg = NetworkDataGenerator()
    filename = (self.dirName +
        "/../../../data/network_data_generator/multi_sample.csv")
    ndg.split(filename, 2, 3, False)
    dataOutputFile = (self.dirName +
        "/../../../data/network_data_generator/multi_sample_split.csv")
    categoriesOutputFile = (self.dirName +
        "/../../../data/network_data_generator/multi_sample_categories.json")
    ndg.saveData(dataOutputFile, categoriesOutputFile)

    # If no error is raised, then the data is in the correct format
    frs = FileRecordStream(dataOutputFile)


if __name__ == "__main__":
  unittest.main()
