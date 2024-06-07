# Turing machine in a Transformer

Let's explore how neural nets can implement computational models.

The highlight will be a weights for a 2N-layer transformer that gets a
description of a Turing machine in the system prompt and the tape in
the prompt, and outputs the state after N steps of computation.

### Contents
[Warm-up: CA in a CNN](#warm-up-ca-in-a-cnn)<br/>
[Turing machines](#turing-machines)<br/>
[Framework: Transformer Circuits](#framework-transformer-circuits)<br/>
[Representation details](#representation-details)<br/>
[High-level design](#high-level-network-structure)<br/>
[Implementation details](#implementation-details)<br/>
[The easy and the hard parts](#the-easy-and-the-hard-parts)

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

## Turing machines

Honestly, Turing machines? Why the heck?

It's a useless model: too convoluted to analyze, too low-level
to write programs in, too different from hardware to implement.

Even Turing in his early papers hand-waved it away when he wanted to prove
anything. They were just a rhetorical device, needed to convince people
in the 1930's who never saw a computer that it was possible to implement
general computation with mechanics or electronics or whatever. I think we
keep it now only to torture first-years.

But still, the Turing machine is the formal definition of computation, so
here we go.

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
define a bilinear form, and attention is strongest for items that are
parallel according to that form. And then the V matrix defines what data
to copy from one to the other.

These will be key in the implementation below.

## Representation details

The embeddings sequence contains the following, in sequence:
* Transition rules for the Turing machine, each in a single vector
* Tape contents, each tape position in a single vector; One of them is
    marked with a "head" feature
* The current machine state is the last vector, the "output" position

In order to be able to make the transformer weights cleaner, we put everything
in a separate dimension:
* Machine state is uses one-hot encoding
* Tape contents uses one-hot encoding
* Transition rules specify the current and next machine state and tape symbol.
    They re-use the dimesions above but also get their own dimensions for
    one-hot encoding of the output state and symbol
* Transition rules also specify head movement, we allow "left", "stay" and
    "right" and allocate 3 dimensions for this
* These "direction" dimensions are also used to denote the machine head position
    and the positions to its left and right
* Positional embeddings are stored in separate dimensions
* 3 dimensions are allocated as a scratch-pad for calculations

In total, if there are $k$ states and $r$ symbols in the alphabet, a Turing
machine is defined by $k \cdot r$ rules. The tape is part of the representation.
If we include a tape of length $m$ and the positional encoding uses $p$
dimensions, the representation is a matrix of dimensions
$$ (k \cdot r + m + 1) \times (2k + 2r + 6 + p). $$

## High-level network structure

The network is made of 2 transformer layers. For simplicity normalization
is elided. Masking isn't used, to allow the head to travel backwards.
The formula of the network is then:

$$
\begin{align*}
x_0 & = \text{rule}_1 \ldots \text{rule}_n \ \text{tape}_1 \ldots \text{tape}_m \ \text{state} \\
x_1 & = x_0 + A_1(x_0) + A_2(x_0) + A_3(x_0) + A_4(x_0) \\
x_2 & = x_1 + FF_1(x_1) \\
x_3 & = x_2 + A_5(x_2) + A_6(x_2) + A_7(x_2) + A_8(x_2) \\
x_4 & = x_3 + FF_2(x_3)
\end{align*}
$$

where each $A_i$ is an attention head.

The two layers can then be repeated as many times as desired to run multiple
steps of the Turing machine.

The first layer marks the rule to apply (the one matching the symbol under
the head and the current state), and marks the positions to the right and
to the left of the head in the tape. This is done by the different components:
* $A_1$ checks if the state dimensions of each item match these dimensions
    in the "current state" position, and writes the result in the scratch pad.
* $A_2$ checks if the symbol dimensions of each item match these dimensions
    in the tape item marked as "head", and writes the result in the scratch pad.
* $A_3$ copies the "head" dimension to the "left" dimension one position to the
    left, skipping "rule" items.
* $A_4$ copies the "head" dimension to the "right" dimension one position to the
    right, skipping "rule" items.
* $FF_1$ calculates the "and" operation between the comparisons, cleans up
    position calculations in the "left" and "right" dimensions, and cleans the
    scratch pad. It also erases the symbol under the head position in the tape.

The second layer copies the new symbol under the head position in the tape,
copies the new state to the "current state" position, and moves the head on
the tape. What each sub-component does:
* $A_5$ erases the current state and the "active rule" mark
* $A_6$ copies the state from the active rule to the "current state" position
* $A_7$ copies the symbol from the active rule to the head position in the tape
* $A_8$ sets the head position of the tape (either promoting left/right to head
    or erasing them and leaving the head marked)
* $FF_2$ does ??? some cleanup (TBD)

## Implementation details

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
