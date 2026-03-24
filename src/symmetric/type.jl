"""
    SymmetricEquilibration{T<:AbstractFloat, M<:AbstractMatrix{T}} <: AbstractEquilibration{T,M}

Equilibration object for symmetric matrices.

Like [`Equilibration`](@ref), but exploits and preserves symmetry throughout:
only a single scaling diagonal `D` is maintained (so `left_scaling === right_scaling`),
and the scaling is applied entry-wise as `A[i,j] *= D_inv[i] * D_inv[j]` using the symmetric
in-place helper `scale_sym!`.

Both dense symmetric matrices (`Symmetric`, `Matrix`) and sparse
(`SparseMatrixCSC`) matrices are supported.  The constructor verifies that the
input is symmetric (or approximately symmetric) and throws an `ArgumentError`
otherwise.

# Constructors
    SymmetricEquilibration(A::AbstractMatrix)

# Fields
- `DAD`: working copy of the matrix (modified in-place during equilibration)
- `D`: accumulated scaling diagonal (same object for left and right)
- `iters`: number of iterations performed
- `is_scaled`: whether equilibration has been applied

# Throws
- `ArgumentError` if the input matrix is not (approximately) symmetric.

# See also
[`Equilibration`](@ref), [`equilibrate!`](@ref), [`equilibrate`](@ref)

# Examples
```julia
A = Symmetric([4.0 1.0; 1.0 3.0])
eq = SymmetricEquilibration(A)
equilibrate!(eq)
D, DAD, _ = left_scaling(eq), scaled_matrix(eq), right_scaling(eq)
```
"""
mutable struct SymmetricEquilibration{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractEquilibration{T,M}
    DAD::M
    D::Diagonal{T,Vector{T}}
    diag_inv::Vector{T}
    diag_inf::Vector{T}
    iter::Int
    is_scaled::Bool
end

SymmetricEquilibration(M::AbstractMatrix{<:Real}) = SymmetricEquilibration(float(M))

function SymmetricEquilibration(A::M) where {T<:AbstractFloat,M<:AbstractMatrix{T}}
    (issymmetric(A) || isapprox(A, A')) || throw(ArgumentError("Matrix is not symmetric."))
    m = size(A, 1)

    return SymmetricEquilibration{T,M}(
        copy(A),
        Diagonal(Vector{T}(undef, m)),
        Vector{T}(undef, m),
        Vector{T}(undef, m),
        0,
        false
    )
end

_build_equilibration(A::Symmetric) = SymmetricEquilibration(A)