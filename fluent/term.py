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

from fluent.cept import Cept
import random


TARGET_SPARSITY = 3.0


class Term():


  def __init__(self):
    self.bitmap   = None
    self.sparsity = None
    self.width    = None
    self.height   = None
    self.cept     = Cept()
    

  def createFromString(self, string, enablePlaceholder=True):
    response = self.cept.getBitmap(string)
    self.bitmap   = response['positions']
    self.sparsity = response['sparsity']
    self.width    = response['width']
    self.height   = response['height']

    if enablePlaceholder and self.sparsity == 0:
      random.seed(string)
      num = self.width * self.height
      bitmap = random.sample(range(num), int(TARGET_SPARSITY * num / 100))
      self.createFromBitmap(bitmap, self.width, self.height)
      random.seed()

    return self


  def createFromBitmap(self, bitmap, width, height):
    self.bitmap = bitmap
    self.width = width
    self.height = height
    self.sparsity = (100.0 * len(bitmap)) / (width*height)
    return self


  def toArray(self):
    array = [0] * self.width * self.height

    for i in self.bitmap:
      array[i] = 1

    return array


  def closestStrings(self):
    if not len(self.bitmap):
      return []

    return [result['term'] for result in
            self.cept.getClosestStrings(self.bitmap)]


  def closestString(self):
    closestStrings = self.closestStrings()

    if not len(closestStrings):
      return ""

    return closestStrings[0]
