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

The `read` tool can read a text document word-by-word, predicting each next word as it goes. You can find it at `tools/read.py`.

Here is an example (after some training):

   => ./tools/read.py data/childrens_stories.txt --checkpoint=cache/model

   Sequence # |     Term # |         Current Term |       Predicted Term
   ----------------------------------------------------------------------
            1 |          1 |                  The |                woods
            1 |          2 |                 Ugly |             duckling
            1 |          3 |             Duckling |
            3 |          1 |                    A |                 duck
            3 |          2 |                 duck |                  the
            3 |          3 |                 made |                  her
            3 |          4 |                  her |                 nest
            3 |          5 |                 nest |                under
            3 |          6 |                under |                 some
            3 |          7 |                 some |               leaves
            3 |          8 |               leaves |                  she
            4 |          1 |                  She |                  sat
            4 |          2 |                  sat |            unpopular
            4 |          3 |                   on |                  the
            4 |          4 |                  the |                 eggs
            4 |          5 |                 eggs |
            4 |          6 |                   to |                 keep
            4 |          7 |                 keep |                 them
            4 |          8 |                 them |                 warm

# Demos

## Fox demo

To run the [Fox demo](http://numenta.org/blog/2013/11/06/2013-fall-hackathon-outcome.html#fox):

    ./tools/read.py data/associations/foxeat.txt -r
