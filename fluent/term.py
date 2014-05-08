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

import os
import json
from fluent.cept import Cept


class Term():


  def __init__(self):
    self.bitmap   = None
    self.sparsity = None
    self.width    = None
    self.cept     = Cept()
    

  def createFromString(self, string):
    response = self.cept.getBitmap(string)      
    self.bitmap   = response['positions']
    self.sparsity = response['sparsity']
    self.width    = response['width']
    return self


  def createFromBitmap(self, bitmap):
    self.bitmap = bitmap
    return self


  def toArray(self):
    array = [0] * 128 * 128

    for i in self.bitmap:
      array[i] = 1

    return array


  def closestString(self):
    if not len(self.bitmap):
      return ""

    closestStrings = self.cept.getClosestStrings(self.bitmap)
    if not len(closestStrings):
      return ""

    return closestStrings[0]['term']
