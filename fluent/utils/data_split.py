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
    """Split the given samples into train/test sets.

    @param samples list of sample elements of any type
    @returns a list of splits where each split is 2-tuple (training, test)
        where each element is a list of elements from samples
    """
    return NotImplementedError()



class KFolds(DataSplit):
  """Implementation of k-folds cross validation algorithm.

  Sample usage:

      data = [
          ("My manager is bad", "management"),
          ("the equipment needs to be replaced", "facilities"),
          ("I'm not getting paid enough", "compensation"),
          ...,
      ]
      kfolds = KFolds(5)
      splits = kfolds.split(data)
      for trainSamples, testSamples in splits:
        results = runExperiment(trainSamples, testSamples)
        ...

  """


  def __init__(self, k):
    if not isinstance(k, int):
      raise TypeError("k must be integer type, not %r" % type(k))
    if k < 2:
      raise ValueError("k must be 2 or greater, not %i" % k)

    self.k = k


  def split(self, samples):
    """Split the given samples into k train/test sets.

    Each train/test split will have len(samples)/k elements in the test set
    and the rest in the train set. Each fold has a distinct, non-overlapping
    test set from the other folds. The samples themselves can be any type.

    @param samples list of sample elements of any type
    @returns a list of splits where each split is 2-tuple (training, test)
        where each element is a list of elements from samples. Each training/
        test pair contains all elements from samples.
    """
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
