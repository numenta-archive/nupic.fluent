# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
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

import numpy

from nupic.encoders.utils import bitsToString



class Encoder(object):
    """
    An encoder converts a value to a sparse distributed representation (SDR).

    The Encoder superclass implements:
    - bitmapToSDR() returns binary SDR of a bitmap
    - bitmapFromSDR() returns the bitmap rep of an SDR
    - pprintHeader() prints a header describing the encoding to the terminal
    - pprint() prints an encoding to the terminal
    - decodedToStr() returns pretty print string of decoded SDR
    
    Methods/properties that must be implemented by subclasses:
    - encode() returns a numpy array encoding the input
    - decode() returns a list of strings representing a decoded SDR
    - getWidth() returns the output width, in bits
    - getDescription() returns a dict describing the encoded output
    
    """


  def encode(self, inputText):
    """
    Encodes inputText and puts the encoded value into the numpy output array,
    which is a 1-D array of length returned by getWidth().
    Note: The numpy output array is reused, so clear it before updating it.
    
    @param inputData      (str)     Data to encode. This should be validated by 
                                    the encoder subclass.
    @param output         (numpy)   1-D array of same length returned by 
                                    getWidth().
    """
    raise NotImplementedError


  def decode(self, encoded):
    """
    Decodes the SDR encoded. See subclass implementation for details; the
    decoding approaches and return objects vary depending on the encoder.
    
    To pretty print the return value from this method, use decodedToStr().
    
    @param encoded        (numpy)     Encoded 1-d array (an SDR).
    """
    raise NotImplementedError


  def getWidth(self):
    """
    Get an encoding's output width in bits. See subclass implementation for 
    details.
    """
    raise NotImplementedError()


  def getDescription(self):
    """
    Get a dictionary describing the encoding. See subclass implementation for
    details.
    """
    raise NotImplementedError()


  def bitmapToSDR(self, bitmap):
    """Convert SDR encoding from bitmap to binary numpy array."""
    sdr = numpy.zeros(self.w)
    for i in self.bitmap:
      sdr[i] = 1
    return sdr


  def bitmapFromSDR(self, sdr):
    """Convert SDR encoding from binary numpy array to bitmap."""
    return numpy.array([i for i in range(len(sdr)) if sdr[i]==1])


  def pprintHeader(self, prefix=""):
    """
    Pretty-print a header that labels the sub-fields of the encoded output.
    This can be used in conjuction with pprint().
    @param prefix printed before the header if specified
    """
    print prefix,
    description = self.getDescription() + [("end", self.getWidth())]
    for i in xrange(len(description) - 1):
      name = description[i][0]
      width = description[i+1][1] - description[i][1]
      formatStr = "%%-%ds |" % width
      if len(name) > width:
        pname = name[0:width]
      else:
        pname = name
      print formatStr % pname,
    print
    print prefix, "-" * (self.getWidth() + (len(description) - 1)*3 - 1)


  def pprint(self, output, prefix=""):
    """
    Pretty-print the encoded output using ascii art.
    @param output to print
    @param prefix printed before the header if specified
    """
    print prefix,
    description = self.getDescription() + [("end", self.getWidth())]
    for i in xrange(len(description) - 1):
      offset = description[i][1]
      nextoffset = description[i+1][1]
      print "%s |" % bitsToString(output[offset:nextoffset]),
    print


  def decodedToStr(self, decodeResults):
    """
    Return a pretty print string representing the return value from decode().
    """

    (fieldsDict, fieldsOrder) = decodeResults

    desc = ''
    for fieldName in fieldsOrder:
      (ranges, rangesStr) = fieldsDict[fieldName]
      if len(desc) > 0:
        desc += ", %s:" % (fieldName)
      else:
        desc += "%s:" % (fieldName)

      desc += "[%s]" % (rangesStr)

    return desc
