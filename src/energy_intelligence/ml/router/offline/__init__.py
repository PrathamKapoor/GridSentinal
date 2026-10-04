"""Offline analysis. Nothing in here may be reached from a deployable routing path.

The separation is structural, not documentary. ``router.oracle`` computes, for every row,
which expert *would have been* the most accurate once the answer is known. That is
oracle information by definition: it is a function of the realised target, so a system
that routed on it would be scoring itself against the answer.

Deployable modules live one level up, in ``router/``. None of them imports anything from
``router.offline``. A test walks the deployable modules' source and asserts it, and a
second test asserts the oracle is not reachable from the production entry points. If
someone later wires the oracle into a live path, the test fails rather than the
forecast quietly becoming excellent.
"""