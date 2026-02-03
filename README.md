<h1 align='center'> jax-bounded-while </h1>
<h2 align="center">Bounded while loop in JAX.</h2>

[![PyPI platforms][pypi-platforms]][pypi-link]
[![PyPI version][pypi-version]][pypi-link]
[![Actions Status][actions-badge]][actions-link]

This is a micro-package, containing the single function `bounded_while_loop`.
</br> Reverse-mode-friendly, bounded `while_loop` implemented via `lax.scan`.

## Installation

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

```bash
pip install jax-bounded-while
```

## Examples

Simple loop over a scalar:

```python
import jax.numpy as jnp
from jax_bounded_while import bounded_while_loop


def cond_fn(x):
    return x < 5


def body_fn(x):
    return x + 1


result = bounded_while_loop(cond_fn, body_fn, jnp.asarray(0), max_steps=10)
print(result)  # Array(5, dtype=int32)
```

PyTree carry (tuple):

```python
import jax.numpy as jnp
from jax_bounded_while import bounded_while_loop


def cond_fn(state):
    x, _ = state
    return x < 3


def body_fn(state):
    x, y = state
    return x + 1, y * 2


result = bounded_while_loop(
    cond_fn, body_fn, (jnp.asarray(0), jnp.asarray(1)), max_steps=5
)
print(result)  # (Array(3, dtype=int32), Array(8, dtype=int32))
```

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/GalacticDynamics/jax-bounded-while/workflows/CI/badge.svg
[actions-link]:             https://github.com/GalacticDynamics/jax-bounded-while/actions
[pypi-link]:                https://pypi.org/project/jax-bounded-while/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/jax-bounded-while
[pypi-version]:             https://img.shields.io/pypi/v/jax-bounded-while

<!-- prettier-ignore-end -->
