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

import unittest
from mock import patch
import os

from fluent.cept import Cept

class TestCept(unittest.TestCase):

  def testAPIKeyPresent(self):
    with patch.dict('os.environ', {'CEPT_API_KEY': 'apikey123'}):
        cept = Cept()

  @patch('os.environ')
  def testExceptionIfAPIKeyNotPresent(self, mockOS):
    with self.assertRaises(Exception) as e:
      cept = Cept()
    self.assertIn("Missing API key.", e.exception)


if __name__ == '__main__':
  unittest.main()
