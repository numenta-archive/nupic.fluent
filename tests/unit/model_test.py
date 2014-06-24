import mock
import os
import pytest
import unittest

from fluent.model import Model
from fluent.term import Term

MODEL_CHECKPOINT_DIR = "/tmp/fluent-test"
MODEL_CHECKPOINT_PKL_PATH  = MODEL_CHECKPOINT_DIR + "/model.pkl"
MODEL_CHECKPOINT_DATA_PATH = MODEL_CHECKPOINT_DIR + "/model.data"

class TestModel(unittest.TestCase):
  def tearDown(self):
    if os.path.exists(MODEL_CHECKPOINT_DATA_PATH):
      os.remove(MODEL_CHECKPOINT_DATA_PATH)

    if os.path.exists(MODEL_CHECKPOINT_PKL_PATH):
      os.remove(MODEL_CHECKPOINT_PKL_PATH)

    if os.path.exists(MODEL_CHECKPOINT_DIR):
      os.rmdir(MODEL_CHECKPOINT_DIR)


  def testLoadWithoutCheckpointDirectory(self):
    model = Model()

    with self.assertRaises(Exception) as e:
      model.load()
    self.assertIn("No checkpoint directory specified", e.exception)


  def testLoadWithoutCheckpointFile(self):
    model = Model(checkpointDir=MODEL_CHECKPOINT_DIR)

    with self.assertRaises(Exception) as e:
      model.load()
    self.assertIn("Could not find checkpoint file", e.exception)


  """
  @mock.patch('os.path')
  def testLoadWithCheckpointFile(self, mock_path):
    model = Model()

    mock_path.exists.return_value = True
    model.load()
    mock_path.exists.assert_called_with("test")
  """


  def testSaveWithoutCheckpointDirectory(self):
    model = Model()

    with self.assertRaises(Exception) as e:
      model.save()
    self.assertIn("No checkpoint directory specified", e.exception)


  def testFeedTermReturnsTerm(self):
    model = Model()
    term = Term().createFromString("test")

    result = model.feedTerm(term)

    assert type(result) == type(term), "Result is not a Term"


if __name__ == '__main__':
  unittest.main()
