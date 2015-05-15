# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014-15, Numenta, Inc.  Unless you have purchased from
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



class Model(object):
  """
  This is the base class for NLP models, where the subclasses implement
  functionalities specific to various NLP problems. When inheriting from this
  class please take note of which methods MUST be overridden, as documented
  below. The model superclass implements:
    - pprint(): prints a description to the terminal
    - ...
  """
  
  def __init__(self,
               dataDir,
               verbosity):
    
    self.dataDir   = dataDir
    self.verbosity = verbosity


  def getText(self):
    raise NotImplementedError


#  def encodeText(self):
#    raise NotImplementedError


  def trainModel(self):
    raise NotImplementedError


  def testModel(self):
    raise NotImplementedError


  def computeError(self):
    raise NotImplementedError


  def printReport(self):
    raise NotImplementedError
