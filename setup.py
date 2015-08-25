# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014-15, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import platform
import sys

from setuptools import setup


def findRequirements():
  """
  Read the requirements.txt file and parse into requirements for setup's
  install_requirements option.
  """
  return [
    line.strip()
    for line in open("requirements.txt").readlines()
    if not line.startswith("#")
  ]


depLinks = []
if "linux" in sys.platform and platform.linux_distribution()[0] == "CentOS":
  depLinks = [ "https://pypi.numenta.com/pypi/nupic",
               "https://pypi.numenta.com/pypi/nupic.bindings" ]

setup(name="Fluent",
      version="0.3",
      description=("A platform for building language/NLP-based "
                   "applications using NuPIC and Cortical.io\'s API."),
      author="Alexander Lavin",
      author_email="alavin@numenta.com",
      url="https://github.com/numenta/nupic.fluent",
      packages=["fluent"],
      install_requires=findRequirements(),
      dependency_links = depLinks,
     )
