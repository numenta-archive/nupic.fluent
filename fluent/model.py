# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have purchased from
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

import numpy
# This is the class corresponding to the C++ optimized Temporal Pooler
from nupic.research.TP10X2 import TP10X2 as TP

from fluent.term import Term



class Model():


  def __init__(self,
               numberOfCols=16384, cellsPerColumn=8,
                initialPerm=0.5, connectedPerm=0.5,
                minThreshold=164, newSynapseCount=164,
                permanenceInc=0.1, permanenceDec=0.0,
                activationThreshold=164,
                pamLength=10,
                checkpointDir=None):

    self.tp = TP(numberOfCols=numberOfCols, cellsPerColumn=cellsPerColumn,
                initialPerm=initialPerm, connectedPerm=connectedPerm,
                minThreshold=minThreshold, newSynapseCount=newSynapseCount,
                permanenceInc=permanenceInc, permanenceDec=permanenceDec,
                
                # 1/2 of the on bits = (16384 * .02) / 2
                activationThreshold=activationThreshold,
                globalDecay=0, burnIn=1,
                checkSynapseConsistency=False,
                pamLength=pamLength)

    self.checkpointDir = checkpointDir
    self.checkpointPath = None
    self._initCheckpoint()


  def _initCheckpoint(self):
    if self.checkpointDir:
      if not os.path.exists(self.checkpointDir):
        os.makedirs(self.checkpointDir)

      self.checkpointPath = self.checkpointDir + "/model.data"


  def canCheckpoint(self):
    return self.checkpointDir != None


  def hasCheckpoint(self):
    return os.path.exists(self.checkpointPath)


  def load(self):
    if not self.checkpointDir:
      raise(Exception("No checkpoint directory specified"))

    if not self.hasCheckpoint():
      raise(Exception("Could not find checkpoint file"))
      
    self.tp.loadFromFile(self.checkpointPath)


  def save(self):
    if not self.checkpointDir:
      raise(Exception("No checkpoint directory specified"))

    self.tp.saveToFile(self.checkpointPath)


  def feedTerm(self, term):
    """ Feed a Term to model, returning next predicted Term """
    tp = self.tp
    array = numpy.array(term.toArray(), dtype="uint32")
    tp.compute(array, enableLearn = True, computeInfOutput = True)

    predictedCells = tp.getPredictedState()
    predictedColumns = predictedCells.max(axis=1)
    
    predictedBitmap = predictedColumns.nonzero()[0].tolist()
    return Term().createFromBitmap(predictedBitmap)
  

  def resetSequence(self):
    self.tp.reset()
