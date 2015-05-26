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

from fluent.models.classification_model import ClassificationModel
from nupic.encoders import RandomDistributedScalarEncoder  ## swap this for CioEncoder in classify_Cio.py
from nupic.engine import Network



class ClassificationModelNetwork(ClassificationModel):

  def __init__(self,
  		dataDir='data',				# Assumes one file each for training and testing
  		encoder='random'
  		kCV=5									# If = 0, no cross-validation

  		networkFileBase='classifiy_randomSDR_net'
  		train=True
  		test=False
  	):

  	self.dataDir = dataDir
  	self.kCV = kCV
  	self.kNNParams = {
		  'distThreshold': 0.000001,
		  'maxCategoryCount': 10,
		  #'distanceMethod': 'rawOverlap',  # Default is Euclidean distance
			}
		self.networkFileBase = networkFileBase

  	# Init random encoder, w/ same dimensions as the Cio encoder
    self.encoder = RandomDistributedScalarEncoder(w=328, n=16384)


	def createNetwork(self):
	  """
	  Set up the following network: Text -> KNNClassifier Region
	  The network is simply an encoder region reading text and passing the output 
	  to a kNN classifier region.
	  """
  	self.net = Network()

  	# Create the two regions.
  	self.net.addRegion("sensor", "py.RecordSensor",
  		json.dumps({"verbosity": self.verbostiy}))
  	# sensor = net.regions["sensor"].getSelf()
  	# sensor.encoder = self.encoder
  	self.net.addRegion("classifier","py.KNNClassifierRegion",
      json.dumps(self.kNNParams))
  	self.sensor = net.regions['sensor']
  	self.classifier = net.regions['classifier']

  	# Link the two regions, w/ an add'l link to send in category labels.
  	self.net.link("sensor", "classifier", "UniformLink", "", 
  		srcOutput="dataOut", destInput="bottomUpIn")
  	self.net.link("sensor", "classifier", "UniformLink", "",
  		srcOutput="categoryOut", destInput="categoryIn")


	def train(self, k, numSamples):
		"""Train the network."""
		# Load text and encode into random SDRs ????????????????????????????????
		sensor.executeCommand([])

		# Train the classifier on training data
		sensor.setParameter('explorer','Flash')  ## ??????????????
	  classifier.setParameter('inferenceMode', 0)
	  classifier.setParameter('learningMode', 1)
	  for i in range(numSamples):
	  	self.net.run(1)
	  	if i%(numSamples/100) == 0:
	  		print "\tTrained on %i samples so far..."

	  networkFile = os.path.join(self.networkFileBase, "_" + k + ".nta")
	  return networkFile


  def test(self, networkFile):
	 	"""Test the network"""
	 	net = Network(savedNetworkFile)  # self?
	  sensor = net.regions['sensor']
	  classifier = net.regions['classifier']

	







	def runExperiment(self):
		"""
		
		Note: if self.kCV = 0 there will be no cross-validation
		"""

		# TAKE FROM CLASSIFY_ANOMALY_RUNNER.PY

		# Split the training data for cross validation.
			# take the 'training' file in self.dataDir and split it into k files; delete


		# loop through k-folds
			numSamples = 0

		pass	
