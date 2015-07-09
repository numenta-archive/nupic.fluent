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

"""Tests for the TextPreprocess class."""

import unittest

from fluent.utils.text_preprocess import TextPreprocess



class TextPreprocessTest(unittest.TestCase):


  def assertOrderedItemsEqual(self, actual, expected):
    """
    Asserts that the two inputs are equal in the order presented
    @param actual           (list/tuple)      Actual ordered group of items
    @param expected         (list/tuple)      Expected ordered group of items
    """
    self.assertIsInstance(actual, type(expected),
        "Expected type {} but got {}".format(type(expected), type(actual)))

    self.assertEqual(len(expected), len(actual),
        "Expected length {} but got {}".format(len(expected), len(actual)))

    for e, p in zip(expected, actual):
      self.assertEqual(e, p, "Expected {} in list but got {}".format(e, p))


  def assertDictEqual(self, actual, expected):
    """
    Asserts that the two dictionaries have all the same keys and values
    @param actual           (dict)      Actual pairs
    @param expected         (dict)      Expected pairs
    """
    self.assertIsInstance(actual, dict,
        "Expected dictionary but got {}".format(type(actual)))
    expectedKeys = expected.keys()
    actualKeys = actual.keys()
    self.assertItemsEqual(actualKeys, expectedKeys)

    for key in expectedKeys:
      self.assertEqual(expected[key], actual[key],
          "Expected value {} for {}, but got {}".format(expected[key], key,
          actual[key]))


  def testTokenizeNoPreprocess(self):
    """
    Tests that none of the preprocessing methods are used
    """
    text = "I can't work at [identifier deleted] if you don't allw me to wfh"
    processor = TextPreprocess()

    expected_tokens = ["i", "can", "t", "work", "at", "identifier", "deleted",
                       "if", "you", "don", "t", "allw", "me", "to", "wfh"]
    tokens = processor.tokenize(text)
    self.assertOrderedItemsEqual(tokens, expected_tokens)


  def testTokenizeRemoveString(self):
    """
    Tests that a provided string is ignored
    """
    text = "I can't work at [identifier deleted] if you don't allw me to wfh"
    processor = TextPreprocess()

    expected_tokens = ["i", "can", "t", "work", "at", "if", "you", "don",
                       "t", "allw", "me", "to", "wfh"]
    tokens = processor.tokenize(text, removeStrings=["[identifier deleted]"])
    self.assertOrderedItemsEqual(tokens, expected_tokens)


  def testTokenizeExpandAbbreviation(self):
    """
    Tests that abbreviations are expanded
    """
    text = "I can't work at [identifier deleted] if you don't allw me to wfh"
    processor = TextPreprocess()

    expected_tokens = ["i", "can", "t", "work", "at", "identifier", "deleted",
                       "if", "you", "don", "t", "allw", "me", "to", "work",
                       "from", "home"]
      
    tokens = processor.tokenize(text, expandAbbr=True)
    self.assertOrderedItemsEqual(tokens, expected_tokens)


  def testTokenizeExpandContraction(self):
    """
    Tests that contractions are expanded
    """
    text = "I can't work at [identifier deleted] if you don't allw me to wfh"
    processor = TextPreprocess()

    expected_tokens = ["i", "can", "not", "work", "at", "identifier", "deleted",
                       "if", "you", "do", "not", "allw", "me", "to", "wfh"]
    tokens = processor.tokenize(text, expandContr=True)
    self.assertOrderedItemsEqual(tokens, expected_tokens)


  def testNoAbbreviationFile(self):
    """
    Ensures a TextPreprocess object can be created even if there is no
    abbreviation file
    """
    try:
      processor = TextPreprocess(abbrCSV="fake_file.csv")
    except IOError():
      self.fail("TextPreprocess constructor raised an exception")


  def testNoContractionFile(self):
    """
    Ensures a TextPreprocess object can be created even if there is no
    contraction file
    """
    try:
      processor = TextPreprocess(contrCSV="fake_file.csv")
    except IOError():
      self.fail("TextPreprocess constructor raised an exception")


  def testReadExpansionFileNoSuffixes(self):
    """
    Tests that TextPreprocess reads csv files correctly
    """
    processor = TextPreprocess()
    abbreviations = processor.readExpansionFile("abbreviations.csv")
    expectedAbbreviations = {"wfh": "work from home"}
    self.assertDictEqual(abbreviations, expectedAbbreviations)

  
  def testReadExpansionFileWithSuffixes(self):
    """
    Tests that TextPreprocess reads csv files correctly and adds suffixes
    """
    processor = TextPreprocess()
    suffixes = ["", "s", "'s"]
    abbreviations = processor.readExpansionFile("abbreviations.csv", suffixes)
    expectedAbbreviations = {"wfh": "work from home",
                             "wfhs": "work from homes",
                             "wfh's": "work from home's"}
    self.assertDictEqual(abbreviations, expectedAbbreviations)


if __name__ == "__main__":
  unittest.main()
