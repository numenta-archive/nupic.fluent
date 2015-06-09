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

import os
import random

from cortipy.cortical_client import CorticalClient
from fluent.encoders.language_encoder import LanguageEncoder



## TODO: add the methods we've run in experiments b/c exp will be run through here, and anything else useful from cortipy.cortical_client.py
class CioEncoder(LanguageEncoder):
  """
  A language encoder using the Cortical.io API.

  The encoder queries the Cortical.io REST API via the cortipy module, which
  returns data in the form of "fingerprints". These representations are
  converted to binary SDR arrays with this Cio encoder.
  """

  def __init__(self, w=128, h=128):
    if 'CORTICAL_API_KEY' not in os.environ:
      print ("Missing CORTICAL_API_KEY environment variable. If you have a "
        "key, set it with $ export CORTICAL_API_KEY=api_key\n"
        "You can retrieve a key by registering for the REST API at "
        "http://www.cortical.io/resources_apikey.html")
      raise Exception("Missing API key.")

    self.apiKey         = os.environ['CORTICAL_API_KEY']
    self.client         = CorticalClient(self.apiKey,
                                         cacheDir=os.join("./cache"))
    self.targetSparsity = 1.0
    self.w              = w  ## Alternatively get dimensions from cortipy client object?
    self.h              = h
    self.n = w*h


  def encode(self, text):
    """
    Encodes the input text w/ a cortipy client. The client returns a
    dictionary of "fingerprint" info, including the SDR bitmap.

    @param  text    (str, list)       If the input is type str, the encoder
                                      assumes it has not yet been tokenized. A
                                      list input will skip the tokenization
                                      step.
    @return         (list)            SDR.
    """
    if isinstance(text, str):
      text = self.client.tokenize(text)

    try:
      encoding = self.client.getBitmap(text)
    except ValueError:
      encoding = self.client.getTextBitmap(text)


    if encoding.sparsity == 0:  ##TODO: test again when/if this happens
      # No fingerprint so fill w/ random bitmap, seeded for each specific term.
      print ("\tThe client returned a bitmap with sparsity=0 for the string "
            "\'%s\', so we'll generate a pseudo-random SDR with the target "
            "sparsity=%0.1f." % (text, self.targetSparsity))
      state = random.getstate()
      random.seed(text)
      num = self.w * self.h
      bitmap = random.sample(range(num), int(self.targetSparsity * num / 100))
      self._createFromBitmap(bitmap, self.w, self.h)
      random.setstate(state)


    return self.client.getSDR(encoding["fingerprint"]["positions"])


  def decode(self, encoding, numTerms=None):
    """
    Converts an SDR back into the most likely word or words.

    By default, the most likely term will be returned. If numTerms is
    specified, then the Cortical.io API will attempt to return that many;
    otherwise the standard is 10. The return value will be a sequence of
    (term, weight) tuples, where higher weights imply the corresponding term
    better matches the encoding.

    @param  encoding        (list)             SDR.
    @param  numTerms        (int)              The max number of terms to
                                               return.
    @return similar         (list)             List of dictionaries, where keys
                                               are terms and likelihood scores.
    """
    # Convert SDR to bitmap, send to cortipy client.
    terms = client.bitmapToTerms(
      super(CioEncoder, self).bitmapFromSDR(encoding))
    # Convert cortipy response to list of tuples (term, weight)
    return [((term["term"], term["score"])) for term in terms]


  ## TODO: redo fields? delete (see line 81 TODO)?
  def _createFromBitmap(self, bitmap, width, height):
    self.bitmap = bitmap
    self.w = width
    self.h = height
    self.sparsity = (100.0 * len(bitmap)) / (width*height)
    return self


  def compare(self, encoding1, encoding2):
    """
    Compare encodings, returning the distances between the SDRs.
    Example return dict:
      {
        "cosineSimilarity": 0.6666666666666666,
        "euclideanDistance": 0.3333333333333333,
        "jaccardDistance": 0.5,
        "overlappingAll": 6,
        "overlappingLeftRight": 0.6666666666666666,
        "overlappingRightLeft": 0.6666666666666666,
        "sizeLeft": 9,
        "sizeRight": 9,
        "weightedScoring": 0.4436476984102028
      }
    """
    # Format input SDRs as Cio fingerprints
    fp1 = {"fingerprint": {"positions":self.bitmapFromSDR(encoding1)}}
    fp2 = {"fingerprint": {"positions":self.bitmapFromSDR(encoding2)}}

    return self.client.compare(fp1, fp2)


  def getWidth(self):
    return self.w


  def getHeight(self):
    return self.h


  def getDescription(self):
    return self.description
