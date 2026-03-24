"""
    AbstractEquilibration{T<:AbstractFloat, M<:AbstractMatrix}

Abstract base type for all equilibration objects.

Subtypes hold a working copy of the matrix together with the accumulated
diagonal scaling factors and bookkeeping state.  The concrete subtypes are
[`Equilibration`](@ref) (general matrices) and
[`SymmetricEquilibration`](@ref) (symmetric matrices).
"""
abstract type AbstractEquilibration{T<:AbstractFloat,M<:AbstractMatrix} end

"""
    scaled_matrix(eq::AbstractEquilibration) -> AbstractMatrix

Return the internally stored (and potentially scaled) matrix `DAD`.

The returned matrix is a reference to the internal buffer; do not mutate it
directly.  After calling [`equilibrate!`](@ref), this is the equilibrated
matrix satisfying `D1 * A_original * D2 ≈ DAD`.
"""
scaled_matrix(eq::AbstractEquilibration) = eq.DAD

"""
    iterations(eq::AbstractEquilibration) -> Int

Return the number of Ruiz iterations performed during the last call to
[`equilibrate!`](@ref).  Returns `0` if equilibration has not been run yet.
"""
iterations(eq::AbstractEquilibration) = eq.iter

"""
    isscaled(eq::AbstractEquilibration) -> Bool

Return `true` if [`equilibrate!`](@ref) has been applied to `eq` and the
result has not been invalidated by a subsequent [`update!`](@ref) call.
"""
isscaled(eq::AbstractEquilibration) = eq.is_scaled

"""
    left_scaling(eq::AbstractEquilibration) -> Diagonal

Return the left scaling diagonal matrix `D1`.

After equilibration the relationship `D1 * A_original * D2 = scaled_matrix(eq)`
holds.  For [`SymmetricEquilibration`](@ref), `left_scaling` and
`right_scaling` return the same object.
"""
left_scaling(eq::AbstractEquilibration) = eq.D1

"""
    right_scaling(eq::AbstractEquilibration) -> Diagonal

Return the right scaling diagonal matrix `D2`.

After equilibration the relationship `D1 * A_original * D2 = scaled_matrix(eq)`
holds.  For [`SymmetricEquilibration`](@ref), `left_scaling` and
`right_scaling` return the same object.
"""
right_scaling(eq::AbstractEquilibration) = eq.D2

"""
    update!(eq::AbstractEquilibration, A::AbstractMatrix)

Replace the matrix stored in `eq` with `A` and reset the equilibration state.

`A` must have the same size as the matrix used to construct `eq`.  After
calling this function, [`isscaled`](@ref) returns `false` and
[`equilibrate!`](@ref) must be called again to obtain fresh scaling factors.

# Throws
- `ArgumentError` if `size(A)` does not match the size of the stored matrix.
"""
function update!(eq::AbstractEquilibration, A::AbstractMatrix)
    size(eq.DAD) == size(A) || throw(ArgumentError("Matrix not of dimension $(size(eq.DAD)): Received $(size(A))"))
    copyto!(eq.DAD, A)
    reset!(eq)
end

"""
    equilibrate(A::AbstractMatrix; tol=1e-3, max_iter=50) -> (D1, DAD, D2)

Functional interface to Ruiz equilibration.

Constructs a temporary equilibration object, runs [`equilibrate!`](@ref), and
returns the triple `(D1, DAD, D2)` where:
- `D1`: left scaling `Diagonal` matrix
- `DAD`: equilibrated matrix
- `D2`: right scaling `Diagonal` matrix

satisfying `D1 * A * D2 ≈ DAD`.

For symmetric matrices wrapped in `Symmetric`, a
[`SymmetricEquilibration`](@ref) is used automatically and `D1 === D2`.

# Arguments
- `A`: the matrix to equilibrate (integer matrices are promoted to `Float64`)

# Keyword arguments
- `tol::AbstractFloat = 1e-3`: convergence tolerance; iterations stop when all
  row/column infinity norms satisfy `|norm - 1| <= tol`
- `max_iter::Int = 50`: maximum number of scaling iterations

# Examples
```julia
A = [4.0 1.0; 2.0 3.0]
D1, DAD, D2 = equilibrate(A)
# D1 * A * D2 ≈ DAD, with all row/col infinity norms ≈ 1
```
"""
function equilibrate(A::AbstractMatrix; kwargs...)
    eq = _build_equilibration(A)
    equilibrate!(eq; kwargs...)
    return left_scaling(eq), scaled_matrix(eq), right_scaling(eq)
end