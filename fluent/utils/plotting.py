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
"""
This file contains plotting tools for NLP experiment results.
"""

import os
import pandas as pd
import plotly.plotly as py

from plotly.graph_objs import *



class PlotNLP():
  """Class to plot evaluation metrics for NLP experiments."""

  def __init__(self,
               apiKey=None,
               username=None,
               experimentName="experiment"):
    # Instantiate API credentials.
    try:
      self.apiKey = apiKey if apiKey else os.environ["PLOTLY_API_KEY"]
    except:
      print ("Missing PLOTLY_API_KEY environment variable. If you have a "
        "key, set it with $ export PLOTLY_API_KEY=api_key\n"
        "You can retrieve a key by registering for the Plotly API at "
        "http://www.plot.ly")
      raise OSError("Missing API key.")
    try:
      self.username = username if username else os.environ["PLOTLY_USERNAME"]
    except:
      print ("Missing PLOTLY_USERNAME environment variable. If you have a "
        "username, set it with $ export PLOTLY_USERNAME=username\n"
        "You can sign up for the Plotly API at http://www.plot.ly")
      raise OSError("Missing username.")

    py.sign_in(self.username, self.apiKey)

    self.experimentName = experimentName


  def getDataFrame(self, dataPath):
    """Get pandas dataframe of the results CSV."""
    try:
      return pd.read_csv(dataPath)
    except IOError("Invalid data path to file"):
      return


  def confusionMatrix(self, data, normalize=True):
    """Plots the confusion matrix of the CSV specified by dataPath."""
    xyzData = self.interpretConfusionMatrixData(data, normalize)

    data = Data([
      Heatmap(
        z=xyzData["z"],
        x=xyzData["x"],
        y=xyzData["y"],
        colorscale='YIGnBu'
      )
    ])

    layout = Layout(
      title='Confusion matrix for ' + self.experimentName,
      xaxis=XAxis(
        title='Predicted label',
        side='top',
        titlefont=Font(
          family='Courier New, monospace',
          size=18,
          color='#7f7f7f'
        )
      ),
      yaxis=YAxis(
        title='True label',
        titlefont=Font(
          family='Courier New, monospace',
          size=18,
          color='#7f7f7f'
        ),
        autorange='reversed'
      ),
      barmode='overlay',
      autosize=True,
      width=1000,
      height=1000,
      margin=Margin(
        l=200,
        r=80,
        b=80,
        t=450
        )
    )

    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig)
    print "Confusion matrix URL: ", plot_url


  def plotCategoryAccuracies():
    """
    """
    ## TODO


  def plotCummulativeAccuracies():
    """
    """
    ## TODO


  @staticmethod
  def interpretConfusionMatrixData(dataFrame, normalize):
    """Parse pandas dataframe into confusion matrix format."""
    labels = dataFrame.columns.values.tolist()[:-1]
    values = map(list, dataFrame.values)
    import pdb; pdb.set_trace()
    for i, row in enumerate(values):
      values[i] = [v/row[-1] for v in row[:-1]] if normalize else row[:-1]
    cm = {"x":labels,
          "y":labels[:-1],
          "z":values[:-1]
          }
    return cm
