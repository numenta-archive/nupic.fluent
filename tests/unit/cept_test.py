#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have purchased from
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
