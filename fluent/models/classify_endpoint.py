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

import numpy
import os

from fluent.models.classification_model import ClassificationModel

from cortipy.cortical_client import CorticalClient



class ClassificationModelEndpoint(ClassificationModel):
  """
  Class to run the survey response classification task with Coritcal.io
  endpoint encodings.

  From the experiment runner, the methods expect to be fed one sample at a time.
  """

  def __init__(self, verbosity=1):
    if 'CORTICAL_API_KEY' not in os.environ:
      print ("Missing CORTICAL_API_KEY environment variable. If you have a "
        "key, set it with $ export CORTICAL_API_KEY=api_key\n"
        "You can retrieve a key by registering for the REST API at "
        "http://www.cortical.io/resources_apikey.html")
      raise Exception("Missing API key.")

    super(ClassificationModelEndpoint, self).__init__(verbosity)

    # Init CorticalClient and Cortical.io encoder; need valid API key (see
    # CioEncoder init for details).
    self.apiKey = os.environ['CORTICAL_API_KEY']
    self.client = CorticalClient(self.apiKey)
    self.positives = {}
    self.categoryBitmaps = {}


  def encodePattern(self, sample):
    """
    Encode an SDR of the input string by querying the Cortical.io API.

    @param sample     (list)            Tokenized sample, where each item is a
                                        string token.
    @return           (string)          Original string
    """
    return " ".join(sample)


  def resetModel(self):
    """Reset the model"""
    self.positives.clear()
    self.categoryBitmaps.clear()


  def trainModel(self, sample, label):
    """
    Train the classifier on the input sample and label.

    @param sample     (string)          Comment
    @param label      (int)             Reference index for the classification
                                        of this sample.
    """
    if label not in self.positives:
      self.positives[label] = []
    self.positives[label].append(sample)

    categoryBitmap = self.client.createClassification(str(label), self.positives[label])["positions"]

    self.categoryBitmaps[label] = categoryBitmap


  def testModel(self, sample):
    """
    Test the kNN classifier on the input sample. Returns the classification most
    frequent amongst the classifications of the sample's individual tokens.
    We ignore the terms that are unclassified, picking the most frequent
    classification among those that are detected.

    @param sample     (numpy.array)     bitmap encoding of the sample.
    @return           (dictionary)      The distances between the sample and the classes
    """

    distances = {}
    for cat, catBitmap in self.categoryBitmaps.iteritems():
      fp1 = {"fingerprint": {"positions": sample}}
      fp2 = {"fingerprint": {"positions": catBitmap}}
      distance[cat] = self.client.compare(fp1, fp2)

    return distances
