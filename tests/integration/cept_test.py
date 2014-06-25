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
from cept_test_data import foxHardCodedResponse

from fluent.cept import Cept


class CeptTest(unittest.TestCase):
  def testSamePositions(self):
     """ Test that the SDR we get from the server hasn't changed,
          using default settings
     """

     cept = Cept()
     httpResponse = cept.getBitmap("fox")

     self.assertEqual(set(httpResponse['positions']),
                      set(foxHardCodedResponse['positions']))


if __name__ == '__main__':
     unittest.main()


