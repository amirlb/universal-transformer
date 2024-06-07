# Turing machine in a Transformer

Let's explore how neural nets can implement computational models.

## Warm-up: CA in a CNN

My favorite universal computational system is
[Rule 110](https://en.wikipedia.org/wiki/Rule_110).
It's so simple, but can encode any computation with reasonable overhead.

Let's implement it as a neural network.

This is a 1-dimenstional cellular automaton with 2 states where the evolution
depends only on the immediate neighbors. This means we can list the 8 cases
to define the system completely:

    Given     ...   ..x   .x.   .xx
    Set to     .     x     x     x

    Given     x..   x.x   xx.   xxx
    Set to     .     x     x     .

There's a neat expression for this:

$$ x_{n} \leftarrow (x_{n} \text{ or } x_{n+1}) \text{ and not } x_{n-1} x_{n} x_{n+1} $$

That is, in the next turn a cell is on if it was on, or the cell to the right
was on, but not all 3 relevant cells were on.

This has a particularly nice representation in terms of common NN operations.
If we define 0 as "off" and any positive number as "on", an OR operation can
be implemented as addition. The "and not" operation is a subtraction, and
the "and" of the three variables is $\max(0, x_{n-1} + x_{n} + x_{n+1} - 2)$, a
rectified linear operation.

This means we can implement the expression as

$$ x_{n} \leftarrow x_{n} + x_{n+1} - \text{ReLU}(x_{n-1} + x_{n} + x_{n+1} - 2) $$

If we try this with the values 0 and 1 for all the variables, we find that
the result is always 0, 1 or 2. In order to be able to chain layers we need
a function that preserves 0 and 1, and takes 2 to 1. For instance

$$ f(x) = x - \text{ReLU}(x - 1) $$

All together, we get the logic in [nn110.py](nn110.py), which is also available
as a [Colab notebook](https://colab.research.google.com/drive/1RqB2T5sJD_8RGIkJTHL3AY0-D04WNH21).

This is a small expression, and it's nice that despite the no-go theorems
about the capabilities of neural networks, it is possible to get this rule
using so few operations. There are still two ReLU gates here, and it may be
possible to achieve universality using only one. Maybe even without adding
multiple filters per cell.

## Framework: Transformer Circuits

Someone in a discussion group said that a transformer is a natual Turing
machine. I wasn't convinced, so I tried to implement a Turing machine in
a transformer.

Since I read the
[Transformer Circuits](https://transformer-circuits.pub/2021/framework/)
framework it's the way I think about transformers, so I started using that.

The part of the paper that stuck the hardest in my mind is induction heads -
this is a feature of the transformer architecture that allows it to search
for any token, or any value from an earlier layer, and refer to the part of
the sequence starting from the found position.

This means that it will be as easy to find weight tensors that implement a
general Turing machine interpreter as it will be to figure out weights for
a specific Turing machine. Probably even easier.

Another idea that was explained very eloquently in the paper is what an
attention head actually does. The K and Q matrices interpreted together
define a bilinear form, and attention is strongest for elements that are
parallel according to that form. And then the V matrix defines what data
to copy from one to the other.

These will be key in the implementation below.

## Implementing any Turing machine: high-level design

The number of dimensions of the encoding depends on the alphabet size and
on the number of states, but not on the machine itself.

The rules defining the machine, as well as the initial tape and state,
are the embedded sequence.

Both the state space and the alphabet use one-hot encoding. The rules
of the Turing machine are one token each, and the tape is one token each.
The machine state is the last token, which means that after the network
is run, the "output" of the transformer is the output token.

In order to support rules, we actually have two copies of the state and
alphabet dimensions. There are additional dimensions for head movement
directions, that are also used to indicate the head position on the tape,
and a few dimenstions that are used as a scratch-pad for calculations.

TBD: show matrix with sequence locations and embedding dimesions

The networks is made of 2 transformer layers. For simplicity normalization
is elided. Masking isn't used, to allow the head to travel backwards.
The formula of the network is then:

    x = rules | tape | initial state
    x += attention1(x) + attention2(x) + attention3(x)
    x += feedforward1(x)
    x += attention4(x) + attention5(x)
    x += feedforward2(x)

The two layers can then be repeated as many times as desired to run multiple
moves of the Turing machine.

TBD: what each layer does
* Layer 1: mark "one left" and "one right" for the current head, mark the rule
    to apply to the current state
* Layer 2: set the character at the head, set the current state, move the head

# Implementation details

TBD

## The easy and the hard parts

TBD:

* Positional encoding - pain if you want to be exact, language applications
    probably benefit from vagueness
* Mandatory participation in all attention heads - kind of a pain for manual
    implementations, is there dropout in real transofrmers?
* Real applications use
    [superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
    to encode multiple data elements in the residual stream, which we avoided.
