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


def readCSV(csvFile, indices=[2,3]):
	"""
	Read in a CSV file w/ the following formatting:
	- one header row
	- one page

	Note: if a given sample has >1 labels, the sample will be repeated, once for
	each label.

	@param csvFile						(str)							File name for the input CSV.
	@param indices						(list)						List of two items, where the first
																							specifies the column index for the
																							data samples, and the second for
																							the labels.
	@return sampleList				(list)						List of str items, one for each
																							sample.
	@return labelList					(list)						List of str items, where each item
																							is the classification label
																							corresponding to the sample at the
																							same index in sampleList.
	"""
	try:
		with open(csvFile) as f:
			reader = csv.reader(f)
			next(reader, None)
			sampleList = []
			labelList = []
			for line in reader:
				for label in line[indices[1]].split(','):
					# May be multiple labels for this sample.
					sampleList.append(line[indices[0]])
					labelList.append(label)
			return sampleList, labelList
	except IOError as e:
		print e
