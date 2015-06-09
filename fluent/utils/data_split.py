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

"""Data splitting is used to partition data into train and test sets."""



class DataSplit(object):
  """Base class for splitting data into train/test partitions."""


  def split(self, samples):
    return NotImplementedError()



class KFolds(DataSplit):
  """Implementation of k-folds cross validation algorithm."""


  def __init__(self, k):
    if not isinstance(k, int):
      raise TypeError("k must be integer type, not %r" % type(k))
    if k < 2:
      raise ValueError("k must be 2 or greater, not %i" % k)

    self.k = k


  def split(self, samples):
    if len(samples) < self.k:
      raise ValueError(
          "Must have as many samples as number of folds %i" % self.k)

    # Aggregate each train/test set to return
    trainTestSplits = []

    # Make sure we have an indexable list
    samples = list(samples)

    numTest = len(samples) / self.k
    for i in xrange(self.k):
      # Determine the range for the test data for this fold
      start = i * numTest
      end = (i + 1) * numTest

      # Split the samples into train and test sets
      testSamples = samples[start:end]
      trainSamples = samples[:start] + samples[end:]

      trainTestSplits.append((trainSamples, testSamples))

    return trainTestSplits
