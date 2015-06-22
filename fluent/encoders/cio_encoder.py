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

import itertools
import os
import random

from cortipy.cortical_client import CorticalClient
from cortipy.exceptions import UnsuccessfulEncodingError
from fluent.encoders.language_encoder import LanguageEncoder



class CioEncoder(LanguageEncoder):
  """
  A language encoder using the Cortical.io API.

  The encoder queries the Cortical.io REST API via the cortipy module, which
  returns data in the form of "fingerprints". These representations are
  converted to binary SDR arrays with this Cio encoder.
  """

  def __init__(self, w=128, h=128, cacheDir="./cache", verbosity=0):
    if 'CORTICAL_API_KEY' not in os.environ:
      print ("Missing CORTICAL_API_KEY environment variable. If you have a "
        "key, set it with $ export CORTICAL_API_KEY=api_key\n"
        "You can retrieve a key by registering for the REST API at "
        "http://www.cortical.io/resources_apikey.html")
      raise OSError("Missing API key.")

    self.apiKey         = os.environ['CORTICAL_API_KEY']
    self.client         = CorticalClient(self.apiKey, cacheDir=cacheDir)
    self.targetSparsity = 5.0
    self.w              = w
    self.h              = h
    self.n              = w*h
    self.verbosity      = verbosity


  def encode(self, text):
    """
    Encodes the input text w/ a cortipy client. The client returns a
    dictionary of "fingerprint" info, including the SDR bitmap.

    @param  text    (str)             A non-tokenized sample of text.
    @return         (dict)            Result from the cortipy client. The bitmap
                                      encoding is at
                                      encoding["fingerprint"]["positions"].
    """
    if not text:
      return None
    try:
      encoding = self.client.getTextBitmap(text)
    except UnsuccessfulEncodingError:
      if self.verbosity > 0:
        print ("\tThe client returned no encoding for the text \'{0}\', so "
               "we'll use the encoding of the token that is least frequent in "
               "the corpus.".format(text))
      encoding = self._subEncoding(text)

    return encoding


  def decode(self, encoding, numTerms=10):
    """
    Converts an SDR back into the most likely word or words.

    By default, the most likely term will be returned. If numTerms is
    specified, then the Cortical.io API will attempt to return that many;
    otherwise the standard is 10. The return value will be a sequence of
    (term, weight) tuples, where higher weights imply the corresponding term
    better matches the encoding.

    @param  encoding        (list)            Bitmap encoding.
    @param  numTerms        (int)             The max number of terms to return.
    @return                 (list)            List of dictionaries, where keys
                                              are terms and likelihood scores.
    """
    terms = client.bitmapToTerms(encoding, numTerms=numTerms)
    # Convert cortipy response to list of tuples (term, weight)
    return [((term["term"], term["score"])) for term in terms]


  def _subEncoding(self, text):
    """
    @param text             (str)             A non-tokenized sample of text.
    @return encoding        (dict)            Fingerprint from cortipy client.
                                              An empty dictionary of the text
                                              could not be encoded.
    """
    tokens = list(itertools.chain.from_iterable(
      [t.split(',') for t in self.client.tokenize(text)]))
    try:
      encoding = min([self.client.getBitmap(t) for t in tokens],
                     key=lambda x: x["df"])
      ## TODO: take union of FPs instead
    except UnsuccessfulEncodingError:
      if self.verbosity > 0:
        print ("\tThe client returned no substitute encoding for the text "
               "\'{0}\', so we encode with None.".format(text))
      encoding = None

    return encoding


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
