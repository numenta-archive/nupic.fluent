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
    if 'CEPT_APP_ID' not in os.environ or 'CEPT_APP_KEY' not in os.environ:
      print("Missing CEPT_APP_ID and CEPT_APP_KEY environment variables.")
      print("You can retrieve these by registering for the CEPT API at ")
      print("https://cept.3scale.net/")
      raise

    self.appId  = os.environ['CEPT_APP_ID']
    self.appKey = os.environ['CEPT_APP_KEY']

    self.client = pycept.Cept(self.appId, self.appKey, cache_dir=CACHE_DIR)


  def getBitmap(self, string):
    return self.client.getBitmap(string)


  def getClosestStrings(self, bitmap, width=128, height=128):
    return self.client.bitmapToTerms(width, height, bitmap)
