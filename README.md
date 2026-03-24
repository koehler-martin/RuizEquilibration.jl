<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="assets/logo.svg">
        <img 
            alt="RuizEquilibration.jl logo"
            src="assets/logo.svg"
            width="200"
            style="border-radius: 16px;">
    </picture>
    <h1>RuizEquilibration.jl</h1>
</div>

[![Build Status](https://github.com/koehler-martin/RuizEquilibration.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/koehler-martin/RuizEquilibration.jl/actions/workflows/CI.yml?query=branch%3Amain)

**RuizEquilibration.jl** provides an implementation of the *Ruiz matrix equilibration* algorithm, an iterative scaling technique introduced by [Daniel Ruiz](https://www.numerical.rl.ac.uk/media/reports/drRAL2001034.pdf) for balancing the magnitudes of rows and columns in a matrix.

Given a matrix $A \in \mathbb{R}^{m \times n}$, the algorithm computes diagonal scaling matrices
$D_1 \in \mathbb{R}^{m \times m}$ and $D_2 \in \mathbb{R}^{n \times n}$
such that the scaled matrix

$$
\widehat{A} = D_1 A D_2
$$

has rows and columns whose infinity norms are close to 1.

This equilibration improves the numerical conditioning of linear systems and optimization problems involving $A$, and is often used as a preprocessing step in numerical linear algebra and convex optimization.

For instance, the scaled system $\widehat{A}\hat{x} = \hat{b}$ is obtained through the transformations $\hat{x} = D_2^{-1} x$ and $\hat{b} = D_1 b$.

Additionally, the scaled quadratic problem

$$
\min_{\hat{x}} \frac{1}{2} \hat{x}^\top \widehat{H} \hat{x} + \hat{x}^\top \hat{h} \quad \text{s.t. } l \leq \widehat{A} \hat{x} \leq u
$$

is obtained through the transformations $\hat{x} = D^{-1} x$, $\widehat{H} = D H D$, $\hat{h} = D h$, and $\widehat{A} = A D$.


## Installation
It can be installed using the Julia package manager by pressing `]` in the REPL and entering:
```julia
https://github.com/koehler-martin/RuizEquilibration.jl
```

## Usage

Both dense matrices and sparse matrices (`SparseMatrixCSC` from `SparseArrays`) are supported.

### Quick start

The `equilibrate` function returns the left scaling, the scaled matrix, and the right scaling directly:

```julia
using RuizEquilibration

A = rand(5, 7)
D1, A_scaled, D2 = equilibrate(A)
# D1 * A * D2 ≈ A_scaled
```

For symmetric matrices, pass a `Symmetric`-wrapped matrix. In that case both scalings are identical (`D1 === D2`):

```julia
S = A * A'
D1, S_scaled, D2 = equilibrate(Symmetric(S))
# D1 === D2, and D1 * S * D1 ≈ S_scaled
```

### In-place interface

When the same matrix structure is equilibrated repeatedly (e.g., in an iterative solver), creating an `Equilibration` instance avoids repeated allocation:

```julia
# --- Nonsymmetric ---
A = rand(5, 7)
eq = Equilibration(A)
equilibrate!(eq; tol=1e-4)

D1 = left_scaling(eq)   # left  diagonal matrix D1
D2 = right_scaling(eq)  # right diagonal matrix D2
A_scaled = scaled_matrix(eq)  # scaled matrix D1 * A * D2
n = iterations(eq)     # number of iterations performed
```

Similarly, for symmetric matrices use `SymmetricEquilibration`, which accepts both plain matrices and `Symmetric`-wrapped ones:

```julia
S = A * A'
eq = SymmetricEquilibration(S)
equilibrate!(eq; tol=1e-4)

D = left_scaling(eq)   # left_scaling(eq) === right_scaling(eq)
S_scaled = scaled_matrix(eq)  # scaled matrix D * S * D
```

### Reusing an equilibration instance

Call `update!` to load a new matrix of the same size into an existing instance, then equilibrate again:

```julia
update!(eq, A_new)
equilibrate!(eq)
```

### Parameters

Both `equilibrate!` and `equilibrate` accept the following keyword arguments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tol` | `AbstractFloat` | `1e-3` | Convergence tolerance. The algorithm stops when the infinity norms of all rows and columns deviate from 1 by less than `tol`. |
| `max_iter` | `Int` | `50` | Maximum number of iterations. The algorithm stops when either convergence is reached or this limit is hit. |

### Accessor functions

| Function | Returns |
|----------|---------|
| `scaled_matrix(eq)` | The scaled matrix $\widehat{A}$ (mutated in place) |
| `left_scaling(eq)` | Left diagonal scaling matrix $D_1$ |
| `right_scaling(eq)` | Right diagonal scaling matrix $D_2$ (equals $D_1$ for symmetric) |
| `iterations(eq)` | Number of iterations performed |
| `isscaled(eq)` | `true` if `equilibrate!` has been called since the last `update!` |

## Acknowledgements

This work is funded by the Deutsche Forschungsgemeinschaft (DFG) under ([Ki 1839/3-2](https://gepris.dfg.de/gepris/projekt/504930816)) and by the European Union through ERC Consolidator Grant SCARCE ([101087662](https://cordis.europa.eu/project/id/101087662)).

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## Reference

Ruiz, D. (2001). *A Scaling Algorithm to Equilibrate Both Rows and Columns Norms in Matrices*. Technical Report RAL-TR-2001-034, Rutherford Appleton Laboratory. Available [here](https://www.numerical.rl.ac.uk/media/reports/drRAL2001034.pdf).
