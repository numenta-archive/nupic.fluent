# # ----------------------------------------------------------------------
# # Numenta Platform for Intelligent Computing (NuPIC)
# # Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# # Numenta, Inc. a separate commercial license for this software code, the
# # following terms and conditions apply:
# #
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License version 3 as
# # published by the Free Software Foundation.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# # See the GNU General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program.  If not, see http://www.gnu.org/licenses.
# #
# # http://numenta.org/licenses/
# # ----------------------------------------------------------------------

# import os
# import time

# from fluent.models.classification_model import ClassificationModel



# class AnomalyScoreClassifier(ClassificationModel):

#   def __init__(self):
#   	self.checkpointDir = None
#     self.dataPath = None
#     self.basePath = None


# 	def train(self):
# 		"""
# 		Create a model for each classification category.

# 		Ass
# 		"""
#   	models = {}
#   	trainDataPath = os.path.join(self.dataPath, "training")

#   	# get data and classes
#   	# inputData = super(AnomalyScoreClassifier, self).CVSplit(trainDataPath)  ## TODO
#   	for samples, classification in inputData:
#   		# Get or create the model for this classification
#     	model = models.get(classification, None)
# 	    if not model:
# 	      model = Model(checkpointDir=os.path.join(checkpointDir, classification))
# 	      models[classification] = model


# 	  n = 0
# 	  start = time.time()
		
