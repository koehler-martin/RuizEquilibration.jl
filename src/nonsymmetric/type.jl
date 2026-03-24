"""
    Equilibration{T<:AbstractFloat, M<:AbstractMatrix{T}} <: AbstractEquilibration{T,M}

Equilibration object for general (rectangular or nonsymmetric) matrices.

Holds a working copy of the matrix along with left and right scaling diagonals
`D1` and `D2` and preallocated work buffers.  After calling
[`equilibrate!`](@ref) the relation `D1 * A_original * D2 ≈ DAD` holds, where
all row and column infinity norms of `DAD` are within the convergence tolerance
of 1.

Both dense (`Matrix`) and sparse (`SparseMatrixCSC`) matrices are supported.
Integer matrices are automatically promoted to `Float64`.

# Constructors
    Equilibration(A::AbstractMatrix)

# Fields
- `DAD`: working copy of the matrix (modified in-place during equilibration)
- `D1`: accumulated left scaling diagonal
- `D2`: accumulated right scaling diagonal
- `iters`: number of iterations performed
- `is_scaled`: whether equilibration has been applied

# See also
[`SymmetricEquilibration`](@ref), [`equilibrate!`](@ref), [`equilibrate`](@ref)

# Examples
```julia
A = [4.0 1.0; 2.0 3.0]
eq = Equilibration(A)
equilibrate!(eq)
D1, DAD, D2 = left_scaling(eq), scaled_matrix(eq), right_scaling(eq)
```
"""
mutable struct Equilibration{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractEquilibration{T,M}
    DAD::M
    D1::Diagonal{T,Vector{T}}
    D2::Diagonal{T,Vector{T}}
    diag_row_inv::Vector{T}
    diag_col_inv::Vector{T}
    row_inf::Vector{T}
    col_inf::Vector{T}
    iter::Int
    is_scaled::Bool
end

Equilibration(A::AbstractMatrix{<:Real}) = Equilibration(float(A))

function Equilibration(A::M) where {T<:AbstractFloat,M<:AbstractMatrix{T}}
    m, n = size(A)
    return Equilibration{T,M}(
        copy(A),
        Diagonal(Vector{T}(undef, m)),
        Diagonal(Vector{T}(undef, n)),
        Vector{T}(undef, m),
        Vector{T}(undef, n),
        Vector{T}(undef, m),
        Vector{T}(undef, n),
        0,
        false
    )
end

_build_equilibration(A::AbstractMatrix) = Equilibration(A)