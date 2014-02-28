# Fluent - Language with NuPIC

A platform for building language / NLP-based applications using [NuPIC](https://github.com/numenta/nupic) and [CEPT](http://www.cept.at/).

## Installation

Requirements:

- [NuPIC](https://github.com/numenta/nupic)
- [pycept](https://github.com/numenta/pycept)

To install, run:

    python setup.py install

Then, set up the following environment variables with your CEPT API credentials:

    export CEPT_APP_ID=app_id
    export CEPT_APP_KEY=app_key

## Usage

### Example

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

### Tool: read

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
             ...

## Demos

### Fox demo

To run the [Fox demo](http://numenta.org/blog/2013/11/06/2013-fall-hackathon-outcome.html#fox):

    => ./tools/read.py data/associations/foxeat.txt -r

    Sequence # |     Term # |         Current Term |       Predicted Term
    ----------------------------------------------------------------------
             1 |          1 |                 frog |
             1 |          2 |                  eat |
             1 |          3 |                flies |
             2 |          1 |                  cow |
             2 |          2 |                  eat |                flies
             2 |          3 |                grain |
             3 |          1 |             elephant |
             3 |          2 |                  eat |                grain
             3 |          3 |               leaves |
             4 |          1 |                 goat |
             4 |          2 |                  eat |                grain
             4 |          3 |                grass |
             5 |          1 |                 wolf |
             5 |          2 |                  eat |                grass
             5 |          3 |               rabbit |
             6 |          1 |                  cat |
             6 |          2 |                likes |
             6 |          3 |                 ball |
             7 |          1 |             elephant |                  eat
             7 |          2 |                likes |
             7 |          3 |                water |
             8 |          1 |                sheep |
             8 |          2 |                  eat |                grass
             8 |          3 |                grass |
             9 |          1 |                  cat |                likes
             9 |          2 |                  eat |
             9 |          3 |               salmon |
            10 |          1 |                 wolf |                  eat
            10 |          2 |                  eat |               rabbit
            10 |          3 |                 mice |
            11 |          1 |                 lion |
            11 |          2 |                  eat |                grass
            11 |          3 |                  cow |                  eat
            12 |          1 |                  dog |
            12 |          2 |                likes |                water
            12 |          3 |                sleep |
            13 |          1 |               coyote |
            13 |          2 |                  eat |                grass
            13 |          3 |                 mice |
            14 |          1 |               coyote |                  eat
            14 |          2 |                  eat |                 mice
            14 |          3 |               rodent |
            15 |          1 |               coyote |                  eat
            15 |          2 |                  eat |               rodent
            15 |          3 |               rabbit |
            16 |          1 |                 wolf |                  eat
            16 |          2 |                  eat |                 mice
            16 |          3 |             squirrel |
            17 |          1 |                  cow |                  eat
            17 |          2 |                  eat |                grain
            ...