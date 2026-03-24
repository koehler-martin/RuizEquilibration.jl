"""
    equilibrate!(eq::Equilibration; tol::AbstractFloat=1e-3, max_iter::Int=50)

Apply Ruiz equilibration in-place to the matrix stored in `eq`.

Iterations stop when all row and column infinity norms satisfy
`|norm - 1| ≤ tol`, or when `max_iter` is reached.

If `eq` is already scaled ([`isscaled`](@ref) returns `true`), this function
returns immediately without performing any work.

# Keyword arguments
- `tol::AbstractFloat = 1e-3`: convergence tolerance
- `max_iter::Int = 50`: maximum number of iterations

# Returns
`nothing` (modifies `eq` in-place).

# See also
[`equilibrate`](@ref), [`Equilibration`](@ref)
"""
function equilibrate!(eq::Equilibration; tol::AbstractFloat=1e-3, max_iter::Int=50)
    A = eq.DAD
    D1 = eq.D1
    D2 = eq.D2
    diag_row_inv = eq.diag_row_inv
    diag_col_inv = eq.diag_col_inv
    row_inf = eq.row_inf
    col_inf = eq.col_inf

    eq.is_scaled && return nothing

    reset!(eq)

    compute_inf_norms!(row_inf, col_inf, A)
    for iter in 1:max_iter
        @. diag_row_inv = ifelse(iszero(row_inf), 1, 1 / sqrt(row_inf))
        @. diag_col_inv = ifelse(iszero(col_inf), 1, 1 / sqrt(col_inf))

        # Scale rows/cols
        lmul!(Diagonal(diag_row_inv), A)
        rmul!(A, Diagonal(diag_col_inv))

        # Update D1 and D2
        @. D1.diag *= diag_row_inv
        @. D2.diag *= diag_col_inv

        compute_inf_norms!(row_inf, col_inf, A)

        eq.iter = iter
        check_termination(row_inf, tol) && check_termination(col_inf, tol) && break
    end
    eq.is_scaled = true
    return nothing
end