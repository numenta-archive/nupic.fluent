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

    self.assertLessEqual(set(("width", "positions", "sparsity", "height")),
                         set(response))

    self.assertIsInstance(response['positions'], list, "Positions field is not a list")
    self.assertIsInstance(response['sparsity'], float, "Sparsity field is not a list")
    self.assertIsInstance(response['width'], int, "Width field is not an int")
    self.assertIsInstance(response['height'], int, "Height field is not an int")


  def testGetClosestStrings(self):
    """ Type check """
    cept = Cept()
    response = cept.getBitmap("snake")

    result = cept.getClosestStrings(response['positions'])
    self.assertTrue(type(result), 'list')


if __name__ == '__main__':
  unittest.main()
