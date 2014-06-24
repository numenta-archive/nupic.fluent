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
import mock
from mock import call

from fluent.term import Term

class TermTest(unittest.TestCase):

  @mock.patch("fluent.cept.Cept.getBitmap")
  def testCreateFromString(self, ceptMock):
    term = Term()
    term = term.createFromString("fox")

    # Check that our mock object was called
    ceptMock.assert_called_with("fox")

    # Check that we have a Term type
    self.assertIsInstance(term, Term)


if __name__ == "__main__":
  unittest.main()
