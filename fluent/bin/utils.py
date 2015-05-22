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
This file contains utility functions to use with nupic.fluent experiments.
"""

import csv
import os

from collections import Counter



exclusions = ('!', '.', ':', ',', '"', '\'', '\n', '?')


def tokenize(string, ignoreCommon=False):
  """
  Tokenize the string into a list of strings, leaving out all chars in the
  exclusions list.
  """
  if not isinstance(string, str):
    raise ValueError("Must input a single string object to tokenize.")
  line = "".join([c for c in string if c not in exclusions])
  strings = line.split(" ")
  if ignoreCommon:
  	common = getFrequentWords()
  	strings = [s for s in strings if s not in common]
  return strings


def getFrequentWords(n=200):
	"""Returns the n most frequent English words/lemmas."""
	if n>1000:
		raise ValueError("Max number of words is 1000.")
	datafile = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data/etc/word_frequencies.txt'))
	try:
		with open(datafile) as f:
			lst = []
			for i, line in enumerate(f):
				if i==0: continue
				if i>99:
					lst.append(line.split(' ')[3])
				elif i>9:
					lst.append(line.split(' ')[4])
				else:
					lst.append(line.split(' ')[5])
				if i==n: break
			return lst
	except IOError:
		print ("Cannot find the word frequencies data file.")


def getCSVInfo(csvFile):
	"""
	Read in a CSV file w/ the following formatting:
	- one header row
	- one page
	- headers: 'index', 'question', 'response', 'classification'

	@param filename					(str)								File name for the input CSV.
	@return labels					(list)
	@return numSamples			(int)
	"""
	try:
		with open(csvFile) as f:
			reader = csv.reader(f)
			next(reader, None)  # skip the headers row
			labels = []
			numSamples = 0
			for line in reader:
				numSamples += 1
				for label in line[3].split(','):
					# may be multiple labels for this line
					if not label in labels:
						labels.append(label)
			return labels, numSamples
	except IOError:
		print ("Input file does not exist.")


def crossValidationSplit(k, counts):  ## Use this for more exact k-folds??
	"""
	Returns indices where the dataset should be split for k-fold validation. That
	is, the partition indices correspond to CSV line number.
	"""
	partitions = []
	splitSize = sum(counts)/k
	split = splitSize
	sampleCount = 0
	for i in range(len(counts)):
		sampleCount += counts[i]
		if sampleCount >= split:
			partitions.append(i)
			split += splitSize
	return partitions


def readCSV(csvFile):
	"""
	Read in a CSV file w/ the following formatting:
	- one header row
	- one page
	- headers: 'index', 'question', 'response', 'classification'

	@param csvFile					(str)								File name for the input CSV.
	@return dataDict				(dict)							Dictionary where a given key is a 
																							classification label, and the 
																							values are a list of tokens.
	"""
	try:
		with open(csvFile) as f:
			reader = csv.reader(f)
			next(reader, None)  # skip the headers row
			dataDict = {}
			for line in reader:
				tokens = tokenize(line[2], ignoreCommon=False)
				for label in line[3].split(','):
					# may be multiple labels for this line
					if not label in dataDict.keys():
						# init the list if this is the first time seeing the label
						dataDict[label] = []
						[dataDict[label].append(t) for t in tokens]
			return dataDict
	except IOError:
		print ("Input file does not exist.")
