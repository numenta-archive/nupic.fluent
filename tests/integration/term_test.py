#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have purchased from
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
import unittest

from fluent.term import Term



class TestTerm(unittest.TestCase):


  def test_createFromString(self):
    # Test enablePlaceholder
    term = Term().createFromString("thisisaninvalidterm", enablePlaceholder=False)
    self.assertEqual(sum(term.toArray()), 0)

    term = Term().createFromString("thisisaninvalidterm", enablePlaceholder=True)
    self.assertGreater(sum(term.toArray()), 0)
    self.assertGreater(term.sparsity, 0)
    placeholder = term.bitmap

    # Make sure we get the same placeholder back for the same term
    term = Term().createFromString("thisisaninvalidterm", enablePlaceholder=True)
    self.assertEqual(term.bitmap, placeholder)

    # Make sure we get a different placeholder back for a different term
    term = Term().createFromString("differentinvalidterm", enablePlaceholder=True)
    self.assertNotEqual(term.bitmap, placeholder)



if __name__ == '__main__':
  unittest.main()
