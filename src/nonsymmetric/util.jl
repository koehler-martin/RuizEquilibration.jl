function reset!(eq::Equilibration)
    fill!(eq.D1.diag, 1)
    fill!(eq.D2.diag, 1)
    fill!(eq.diag_row_inv, 1)
    fill!(eq.diag_col_inv, 1)
    fill!(eq.row_inf, 0)
    fill!(eq.col_inf, 0)
    eq.iter = 0
    eq.is_scaled = false
end