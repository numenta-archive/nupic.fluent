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

import os

try:
  import simplejson as json
except ImportError:
  import json



class Model(object):
  """
  This is the base class for NLP models, where the subclasses implement
  functionalities specific to various NLP problems. When inheriting from this
  class please take note of which methods MUST be overridden, as documented
  below.

  The Model superclass implements:
    - getEncoderDescription()
    - pprint(): prints a description to the terminal
    - ...

  Methods/properties that must be implemented by subclasses:

  """
  
  # def __init__(self,
  #             checkpointDir,
  #             dataDir,
  #             resultsDir,
  #             verbostiy)

  #   self.checkpointDir        = checkpointDir
  #   self.checkpointDataPath   = None
  #   self.checkpointJsonPath   = None
  #   self.dataDir              = dataDir
  #   self.getEncoder           = None           ####### ???????????
  #   self.resultsDir           = resultsDir
  #   self.verbosity            = verbosity

  #   self._initCheckpoint()


  def getInfo(self):
    """Return info about the model."""


    return {}


  # def _initCheckpoint(self):
  #   if self.checkpointDir:
  #     if not os.path.exists(self.checkpointDir):
  #       os.makedirs(self.checkpointDir)
  #     self.checkpointDataPath = self.checkpointDir + "/model.data"
  #     self.checkpointJsonPath = self.checkpointDir + "/model.json"
    

  # def canCheckpoint(self):
  #   return self.checkpointDir != None


  # def hasCheckpoint(self):
  #   return (os.path.exists(self.checkpointJsonPath) and
  #           os.path.exists(self.checkpointDataPath))


  # def save(self):
  #   """Save the model."""
  #   if not self.checkpointDir:
  #     raise(Exception("No checkpoint directory specified"))

  #   self.tp.saveToFile(self.checkpointDataPath)

  #   with open(self.checkpointJsonPath, 'wb') as f:
  #     json.dump(self.tp, f)


  # def load(self):
  #   """Load the model in the checkpoint path."""
  #   if not self.checkpointDir:
  #     raise(Exception("No checkpoint directory specified"))

  #   if not self.hasCheckpoint():
  #     raise(Exception("Could not find checkpoint file"))
      
  #   with open(self.checkpointJsonPath, 'rb') as f:
  #     self.tp = json.load(f)

  #   self.tp.loadFromFile(self.checkpointDataPath)







  def trainModel(self):
    raise NotImplementedError


  def testModel(self):
    raise NotImplementedError


  def evaluateResults(self):
    raise NotImplementedError


  def printReport(self): ## TODO
    # raise NotImplementedError
    pass
