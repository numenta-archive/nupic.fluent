import unittest

from fluent.cept import Cept

class TestCept(unittest.TestCase):

  def testGetBitmap(self):
    """ Type check what we get back from the cept object """
    cept = Cept()
    response = cept.getBitmap("fox")

    self.assertTrue(type(response), 'dict')
    self.assertTrue(type(response['positions']), 'list')
    self.assertTrue(type(response['sparsity']), 'float')
    self.assertTrue(type(response['width']), 'int')
    self.assertTrue(type(response['height']), 'int')


  def testGetClosestStrings(self):
    """ Type check """
    cept = Cept()
    response = cept.getBitmap("snake")

    result = cept.getClosestStrings(response['positions'])
    self.assertTrue(type(result), 'list')


if __name__ == '__main__':
  unittest.main()
