left_scaling(eq::SymmetricEquilibration)  = eq.D
right_scaling(eq::SymmetricEquilibration) = eq.D

function reset!(eq::SymmetricEquilibration)
    fill!(eq.D.diag, 1)
    fill!(eq.diag_inv, 1)
    fill!(eq.diag_inf, 0)
    eq.iter = 0
    eq.is_scaled = false
end