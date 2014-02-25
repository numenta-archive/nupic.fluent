# Fluent - Language with NuPIC

A platform for building language / NLP-based demos using [NuPIC](https://github.com/numenta/nupic) and [CEPT](http://www.cept.at/).

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
