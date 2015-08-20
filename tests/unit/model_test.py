#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import os
import sys
from mock import patch, MagicMock, Mock
import pytest
import unittest

from fluent.term import Term

MODEL_CHECKPOINT_DIR = "/tmp/fluent-test"
MODEL_CHECKPOINT_PKL_PATH  = MODEL_CHECKPOINT_DIR + "/model.pkl"
MODEL_CHECKPOINT_DATA_PATH = MODEL_CHECKPOINT_DIR + "/model.data"

class TestModel(unittest.TestCase):

  def setUp(self):
    self.nupicMock = MagicMock()
    modules = {
      'nupic': self.nupicMock,
      'nupic.research': self.nupicMock.research,
      'nupic.research.TP10X2': self.nupicMock.research.TP10X2,
    }
    self.module_patcher = patch.dict('sys.modules', modules)
    self.module_patcher.start()

    from fluent.model import Model
    self.Model = Model


  def tearDown(self):
    self.module_patcher.stop()

    if os.path.exists(MODEL_CHECKPOINT_DATA_PATH):
      os.remove(MODEL_CHECKPOINT_DATA_PATH)

    if os.path.exists(MODEL_CHECKPOINT_PKL_PATH):
      os.remove(MODEL_CHECKPOINT_PKL_PATH)

    if os.path.exists(MODEL_CHECKPOINT_DIR):
      os.rmdir(MODEL_CHECKPOINT_DIR)


  def testLoadWithoutCheckpointDirectory(self):
    model = self.Model()

    with self.assertRaises(Exception) as e:
      model.load()
    self.assertIn("No checkpoint directory specified", e.exception)


  def testLoadWithoutCheckpointFile(self):
    model = self.Model(checkpointDir=MODEL_CHECKPOINT_DIR)

    with self.assertRaises(Exception) as e:
      model.load()
    self.assertIn("Could not find checkpoint file", e.exception)


  def testSaveWithoutCheckpointDirectory(self):
    model = self.Model()

    with self.assertRaises(Exception) as e:
      model.save()
    self.assertIn("No checkpoint directory specified", e.exception)

  @patch.dict('os.environ', {'CEPT_API_KEY': 'testkey123'})
  @patch('nupic.research.TP10X2.TP10X2.compute')
  @patch('pycept.Cept.getBitmap')
  def testFeedTermReturnsTerm(self, mockBitmap, mockTPCompute):
    model = self.Model()
    term = Term().createFromString("test")

    result = model.feedTerm(term)

    self.assertIsInstance(result, Term, "Result is not a Term")


if __name__ == '__main__':
  unittest.main()
