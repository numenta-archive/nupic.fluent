# Fluent - Language with NuPIC

A platform for building language / NLP-based applications using [NuPIC](https://github.com/numenta/nupic) and [CEPT](http://www.cept.at/).

# Installation

Requirements:

- [NuPIC](https://github.com/numenta/nupic)
- [pycept](https://github.com/numenta/pycept)

# Usage

## Example

    from fluent.model import Model
    from fluent.term import Term

    model = Model()

    term1 = Term().createFromString("coyote")
    term2 = Term().createFromString("eats")
    term3 = Term().createFromString("mouse")

    # Train
    for _ in range(3):
      model.feedTerm(term1)
      model.feedTerm(term2)
      model.feedTerm(term3)
      model.resetSequence()

    # Test
    term4 = Term().createFromString("wolf")

    model.feedTerm(term4)
    prediction = model.feedTerm(term2)

    print prediction.closestString()
    # => "mouse"

## Tool: read

The `read` tool is useful for reading a text document word-by-word, predicting each next word as it goes. You can find it at `tools/read.py`.

# Demos

## Fox demo

To run the [Fox demo](http://numenta.org/blog/2013/11/06/2013-fall-hackathon-outcome.html#fox):

    ./tools/read.py data/associations/foxeat.txt -r
