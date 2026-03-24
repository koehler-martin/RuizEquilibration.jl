"""
    equilibrate!(eq::SymmetricEquilibration; tol::AbstractFloat=1e-3, max_iter::Int=50)

Apply Ruiz equilibration in-place to the symmetric matrix stored in `eq`.

Because the scaling is applied symmetrically, the result remains symmetric.
Iterations stop when all column infinity norms satisfy `|norm - 1| ≤ tol`,
or when `max_iter` is reached.

If `eq` is already scaled ([`isscaled`](@ref) returns `true`), this function
returns immediately without performing any work.

# Keyword arguments
- `tol::AbstractFloat = 1e-3`: convergence tolerance
- `max_iter::Int = 50`: maximum number of iterations

# Returns
`nothing` (modifies `eq` in-place).

# See also
[`equilibrate`](@ref), [`SymmetricEquilibration`](@ref)
"""
function equilibrate!(eq::SymmetricEquilibration; tol::AbstractFloat=1e-3, max_iter::Int=50)
    A = eq.DAD
    D = eq.D
    diag_inv = eq.diag_inv
    diag_inf = eq.diag_inf

    eq.is_scaled && return nothing

    reset!(eq)

    compute_col_inf_norms!(diag_inf, A)

    for iter in 1:max_iter
        @. diag_inv = ifelse(iszero(diag_inf), 1, 1 / sqrt(diag_inf))

        # Scale Matrix
        scale_sym!(A, diag_inv)

        # Update D
        @. D.diag *= diag_inv

        compute_col_inf_norms!(diag_inf, A)

        eq.iter = iter
        check_termination(diag_inf, tol) && break
    end
    eq.is_scaled = true
    return nothing
end