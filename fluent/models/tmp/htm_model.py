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

import math
import numpy
import os
import pickle

from fluent.models.model import Model
# This is the class corresponding to the C++ optimized Temporal Pooler
## TO DO: switch this to new implementation temporal_memory.py (or temporal_pooler.py in nupic.research?)
from nupic.research.TP10X2 import TP10X2 as TP

from fluent.term import Term



class HTMModel(Model):


  def __init__(self,
               numberOfCols=64*64, cellsPerColumn=8,
                initialPerm=0.5, connectedPerm=0.5,
                minThreshold=12, newSynapseCount=12,
                permanenceInc=0.1, permanenceDec=0.0,
                activationThreshold=12,
                pamLength=3,
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



  def feedTerm(self, term, learn=True):
    """ Feed a Term to model, returning next predicted Term """
    tp = self.tp
    array = numpy.array(term.toArray(), dtype="uint32")
    tp.compute(array, enableLearn = learn, computeInfOutput = True)

    predictedCells = tp.getPredictedState()
    predictedColumns = predictedCells.max(axis=1)
    
    predictedBitmap = predictedColumns.nonzero()[0].tolist()
    return Term().createFromBitmap(predictedBitmap,
                                   width=term.width,
                                   height=term.height)
  

  def resetSequence(self):
    self.tp.reset()
