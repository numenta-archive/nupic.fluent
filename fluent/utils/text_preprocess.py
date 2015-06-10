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
This file contains text pre-processing functions for NLP experiments.
"""

import os
import re
import string

from collections import Counter



class TextPreprocess(object):
  """Class for text pre-processing"""

  alphabet = string.ascii_lowercase

  def __init__(self,
               corpusTxt='compilation.txt'):

    # compilation.txt is a compilation of most frequent words from both British
    # National Corpus and Wiktionary, and books from Project Guttenberg.

    self.corpusPath = os.path.abspath(os.path.join(
      os.path.dirname(__file__), '../..', 'data/etc', corpusTxt))

    try:
      self.corpus = file(self.corpusPath).read()
    except IOError("Could not open corpus text."):
      raise

    # English language initializations:
    self.txtCorpus = self.tokenize(self.corpus)
    self.bagOfWords = Counter(self.txtCorpus)


  def tokenize(self, text,
               ignoreCommon=None, removeStrings=[], correctSpell=False):
    """
    Tokenize, returning only lower-case letters and "$". Apostrophes are
    deleted; e.g. "didn't" becomes "didnt".
    @param text               (str)             Single string to tokenize.
    @param ignoreCommon       (int)             This many most frequent words
                                                will be filtered out from the
                                                returned tokens.
    @param removeStrings      (list)            List of strings to delete from
                                                the text.
    @param correctSpell       (boolean)         Run tokens through spelling
                                                correction.
    """
    if not isinstance(text, str):
      raise ValueError("Must input a single string object to tokenize.")
    removeStrings.append("'")
    for removal in removeStrings:
      text = text.replace(removal, "")
    tokens = re.findall('[a-z$]+', text.lower())
    if correctSpell:
      tokens = [self.correct(t) for t in tokens]
    if ignoreCommon:
      tokens = self.removeMostCommon(tokens, n=ignoreCommon)
    return tokens


  def removeMostCommon(self, tokenList, n=100):
    """
    Remove the n most common tokens as counted in the bag-of-words corpus.

    @param tokenList        (list)              List of token strings.
    @param n                (int)               Will filter out the n-most
                                                frequent terms.
    """
    ignoreList = [word[0] for word in self.bagOfWords.most_common(n)]
    return [token for token in tokenList if token not in ignoreList]


  def correct(self, word):
    """
    Find the best spelling correction for this word. Prefer edit distance  of 0,
    then one, then two; otherwise default to the word itself.
    """
    candidates = (self._known({word}) or
                  self._known(self._editDistance1(word)) or
                  self._known(self._editDistance2(word)) or
                  [word])

    return max(candidates, key=self.bagOfWords.get)


  def _known(self, words):
    """Return the subset of words that are in the corpus."""
    return {w for w in words if w in self.bagOfWords}


  def _editDistance1(self, word):
    """
    Return all strings that are edit distance =1 from the input word.
    Damerau-Levenshtein edit distance:
    - deletion(x,y) is the count(xy typed as x)
    - insertion(x,y) is the count(x typed as xy)
    - substitution(x,y) is the count(x typed as y)
    - transposition(x,y) is the count(xy typed as yx)
    """
    # First split the word into tuples of all possible pairs.
    # Note: need +1 so we can perform edits at front and back end of the word.
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]

    # Now perform the edits at every possible split location.
    # Substitution is essentially a deletion and insertion.
    delete = [a+b[1:] for a,b in splits if b]
    insert = [a+b+c for a,b in splits for c in TextPreprocess.alphabet]
    subs = [a+c+b[1:] for a,b in splits for c in TextPreprocess.alphabet if b]
    trans = [a+b[1]+b[0]+b[2:] for a,b in splits if len(b)>1]

    return set(delete + insert + subs + trans)


  def _editDistance2(self, word):
    """
    Return all strings that are edit distance =2 from the input word; i.e. call
    the _editDistance1() method twice for edits with distances of two.
    """
    return {edits2 for edits1 in self._editDistance1(word)
            for edits2 in self._editDistance1(edits1)}
