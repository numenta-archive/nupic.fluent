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

import pycept

CACHE_DIR = "./cache"


class Cept():


  def __init__(self):
    if 'CEPT_API_KEY' not in os.environ:
      print("Missing CEPT_API_KEY environment variable.")
      print("You can retrieve this by registering for the CEPT API at ")
      print("http://cept.github.io/CEPT-Website/developers_apikey.html")
      raise Exception("Missing API key.")

    self.apiKey  = os.environ['CEPT_API_KEY']

    self.client = pycept.Cept(self.apiKey, cacheDir=CACHE_DIR)


  def getBitmap(self, string):
    return self.client.getBitmap(string)


  def getClosestStrings(self, bitmap):
    return self.client.bitmapToTerms(bitmap)
