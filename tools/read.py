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

from optparse import OptionParser

from fluent.model import Model
from fluent.term import Term



def readFile(filename):
  exclusions = ('!', '.', ':', ',', '"', '\'', '\n')

  with open(filename) as f:
    for line in f:
      line = "".join([c for c in line if c not in exclusions])
      strings = line.split(" ")

      model = Model()

      for string in strings:
        if not len(string):
          continue

        term = Term().createFromString(string)
        prediction = model.feedTerm(term)

        print("%16s | %20s" % (string, prediction.closestString()))



if __name__ == '__main__':
  parser = OptionParser("%prog file [options]")
  (options, args) = parser.parse_args()

  if not len(args):
    raise(Exception("file required"))

  readFile(args[0])
