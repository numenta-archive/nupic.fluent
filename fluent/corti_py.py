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
"""
nupic.fluent
This houses functions, to run in nupic.fluent, that require calls to the
cortical.io REST API.
"""

import os

from cortipy.cortical_client import CorticalClient



class CortiPy():


  def __init__(self):
    if 'CORTICAL_API_KEY' not in os.environ:
      print ("Missing REST_API_KEY environment variable. If you have a key, "
        "set it with $ export REST_API_KEY=api_key\n"
        "You can retrieve a key by registering for the REST API at "
        "http://www.cortical.io/resources_apikey.html")
      raise Exception("Missing API key.")

    self.apiKey  = os.environ['CORTICAL_API_KEY']

    self.client = CorticalClient(self.apiKey, cacheDir="./cache")


  def getBitmap(self, string):
    # Use cortipy to query the the sdr for a string
    return self.client.getBitmap(string)


  def getClosestStrings(self, bitmap):
    return self.client.bitmapToTerms(bitmap)
