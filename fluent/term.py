# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014-15, Numenta, Inc.  Unless you have purchased from
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
import random

from cortipy.cortical_client import CorticalClient


TARGET_SPARSITY = 1.0

def termJSONEncoder(obj):
  """
  For encoding as JSON. Usage:
    json.dumps(term, default = termJSONEncoder)
  """
  d = {'positions': obj.bitmap,
       'sparsity': obj.sparsity,
       'width': obj.width,
       'height': obj.height}
  return d


def termJSONDecoder(d):
  """
  For decoding from JSON. Usage:
    json.loads(str, object_hook = termJSONDecoder)
  """
  t = Term()
  t.createFromBitmap(d['positions'], d['width'], d['height'])
  return t


class Term():


  def __init__(self):
    if 'CORTICAL_API_KEY' not in os.environ:
      print ("Missing CORTICAL_API_KEY environment variable. If you have a "
        "key, set it with $ export CORTICAL_API_KEY=api_key\n"
        "You can retrieve a key by registering for the REST API at "
        "http://www.cortical.io/resources_apikey.html")
      raise Exception("Missing API key.")

    self.apiKey   = os.environ['CORTICAL_API_KEY']
    self.client   = CorticalClient(self.apiKey, cacheDir="./cache")  # TO DO: name cache by experiment
    
    self.bitmap   = None
    self.sparsity = None
    self.width    = None
    self.height   = None


  def __repr__(self):
    return termJSONEncoder(self)

  
  def createFromString(self, string, enablePlaceholder=True):
    response = self.client.getBitmap(string)
    self.bitmap   = response['fingerprint']['positions']
    self.sparsity = response['sparsity']
    self.width    = response['width']
    self.height   = response['height']

    if enablePlaceholder and self.sparsity == 0:
      # No fingerprint so fill w/ random bitmap, seeded for each specific term.
      print ("\tThe client returned a bitmap with sparsity=0 for the string "
            "\'%s\', so we'll generate a pseudo-random SDR with the target "
            "sparsity=%0.1f." % (string, TARGET_SPARSITY))
      state = random.getstate()
      random.seed(string)
      num = self.width * self.height
      bitmap = random.sample(range(num), int(TARGET_SPARSITY * num / 100))
      self.createFromBitmap(bitmap, self.width, self.height)
      random.setstate(state)

    return self


  def createFromBitmap(self, bitmap, width, height):
    self.bitmap = bitmap
    self.width = width
    self.height = height
    self.sparsity = (100.0 * len(bitmap)) / (width*height)
    return self


  def compare(self, term):
    """
    Compare self with the provided term. Calls REST compare and returns the
    corresponding dict.
    """
    return self.client.compare(self.bitmap, term.bitmap)
  

  def toArray(self):
    array = [0] * self.width * self.height

    for i in self.bitmap:
      array[i] = 1

    return array


  def closestStrings(self):
    if not len(self.bitmap):
      return []
    return [result['term'] for result in
            self.client.bitmapToTerms(self.bitmap)]


  def closestString(self):
    closestStrings = self.closestStrings()

    if not len(closestStrings):
      return ""

    return closestStrings[0]
