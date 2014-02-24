#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have purchased from
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

from fluent.model import Model
from fluent.term import Term



class TestModel(unittest.TestCase):


  def test_training(self):
    term1 = Term().createFromString("fox")
    term2 = Term().createFromString("eats")
    term3 = Term().createFromString("rodent")

    model = Model()

    for _ in range(5):
      model.feedTerm(term1)
      model.feedTerm(term2)
      model.feedTerm(term3)
      model.resetSequence()

    model.feedTerm(term1)
    prediction = model.feedTerm(term2)

    self.assertEqual(prediction.closestString(), "rodent")



if __name__ == '__main__':
  unittest.main()