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
import time

from fluent.encoders.cio_encoder import CioEncoder
from fluent.models.model import Model



class PredictionModel(Model):

def __init__(self,
              numberOfCols=64*64, cellsPerColumn=8,
              initialPerm=0.5, connectedPerm=0.5,
              minThreshold=12, newSynapseCount=12,
              permanenceInc=0.1, permanenceDec=0.0,
              activationThreshold=12,
              pamLength=3,
              checkpointDir=None,
              dataDir=None,
              verbosity=0):

    self.tp = TP(numberOfCols=numberOfCols, cellsPerColumn=cellsPerColumn,
                initialPerm=initialPerm, connectedPerm=connectedPerm,
                minThreshold=minThreshold, newSynapseCount=newSynapseCount,
                permanenceInc=permanenceInc, permanenceDec=permanenceDec,
                activationThreshold=activationThreshold,
                globalDecay=0, burnIn=1,
                checkSynapseConsistency=False,
                pamLength=pamLength)
    


  	if encoder:
	    self.encoder = encoder
	  else:
	  	self.encoder = CioEncoder()




  def trainModel(self):
    """
    """

    pass

  def testModel(self):
    """
    """

    pass






  def feedText(self):  ## NEEDS REWORK B/C NO LONGER USING TERM()
    """ Feed a Term to model, returning next predicted Term """
    tp = self.tp
    array = numpy.array(term.toArray(), dtype="uint32")
    tp.compute(array, enableLearn = learn, computeInfOutput = True)

    predictedCells = tp.getPredictedState()
    predictedColumns = predictedCells.max(axis=1)
    
    predictedBitmap = predictedColumns.nonzero()[0].tolist()
    return Term().createFromBitmap(predictedBitmap, ###############
                                   width=term.width,
                                   height=term.height)


  def computeAnomalyScore(self):
    """
    Return the current anomaly score by comparing previously predicted
    columns with currently active columns.
    """
    tp = self.tp

    # Indices of previously predicted columns
    predictedCells = tp.infPredictedState['t-1']
    predictedColumns = predictedCells.max(axis=1).nonzero()[0]

    # Indices of currently active columns
    activeCells = tp.infActiveState['t']
    activeColumns = activeCells.max(axis=1).nonzero()[0]

    return anomaly.computeAnomalyScore(activeColumns, predictedColumns)


  def resetSequence(self):
    self.tp.reset()
