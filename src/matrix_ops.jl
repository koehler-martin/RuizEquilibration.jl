function check_termination(diag_inf, tol)
    @inbounds for r in diag_inf
        abs(1 - r) > tol && return false
    end
    return true
end

function compute_inf_norms!(row_norms, col_norms, A::AbstractMatrix)
    fill!(row_norms, 0)
    fill!(col_norms, 0)
    @inbounds for j in axes(A, 2)
        col_max = zero(eltype(A))
        @simd for i in axes(A, 1)
            aij = abs(A[i, j])
            row_norms[i] = max(row_norms[i], aij)
            col_max = max(col_max, aij)
        end
        col_norms[j] = col_max
    end
    @. row_norms = ifelse(iszero(row_norms), 1, row_norms)
    @. col_norms = ifelse(iszero(col_norms), 1, col_norms)
    return row_norms, col_norms
end

function compute_inf_norms!(row_norms, col_norms, A::SparseMatrixCSC)
    fill!(row_norms, 0)
    fill!(col_norms, 0)
    rv = rowvals(A)
    nzv = nonzeros(A)
    @inbounds for j in axes(A, 2)
        col_max = zero(eltype(A))
        for k in nzrange(A, j)
            aij = abs(nzv[k])
            row_norms[rv[k]] = max(row_norms[rv[k]], aij)
            col_max = max(col_max, aij)
        end
        col_norms[j] = col_max
    end
    @. row_norms = ifelse(iszero(row_norms), 1, row_norms)
    @. col_norms = ifelse(iszero(col_norms), 1, col_norms)
    return row_norms, col_norms
end

function compute_row_inf_norms!(norms, A::AbstractMatrix)
    fill!(norms, zero(eltype(A)))
    @inbounds for j in axes(A, 2)
        @inbounds for i in axes(A, 1)
            norms[i] = max(norms[i], abs(A[i, j]))
        end
    end
    @. norms = ifelse(iszero(norms), 1, norms)
end

function compute_col_inf_norms!(norms, A::AbstractMatrix)
    fill!(norms, zero(eltype(A)))
    @inbounds for i in axes(A, 1)
        @inbounds for j in axes(A, 2)
            norms[j] = max(norms[j], abs(A[i, j]))
        end
    end
    @. norms = ifelse(iszero(norms), 1, norms)
end

function compute_col_inf_norms!(norms, A::SparseMatrixCSC)
    fill!(norms, 0)
    nzv = nonzeros(A)
    @inbounds for j in axes(A, 2)
        col_max = zero(eltype(A))
        for k in nzrange(A, j)
            col_max = max(col_max, abs(nzv[k]))
        end
        norms[j] = col_max
    end
    @. norms = ifelse(iszero(norms), 1, norms)
    return norms
end

function scale_sym!(A::Symmetric, d)
    M = parent(A)
    if A.uplo == 'U'
        @inbounds for j in axes(M, 2)
            dj = d[j]
            @inbounds for i in 1:j
                M[i, j] *= d[i] * dj
            end
        end
    else
        @inbounds for j in axes(M, 2)
            dj = d[j]
            @inbounds for i in j:size(M, 1)
                M[i, j] *= d[i] * dj
            end
        end
    end
end

function scale_sym!(A::AbstractMatrix, d)
    @inbounds for j in axes(A, 2)
        dj = d[j]
        @inbounds for i in 1:j
            A[i, j] *= d[i] * dj
        end
        @inbounds for i in 1:j-1
            A[j, i] = A[i, j]
        end
    end
end

function scale_sym!(A::SparseMatrixCSC, d)
    rv = rowvals(A)
    nzv = nonzeros(A)
    @inbounds for j in axes(A, 2)
        dj = d[j]
        for k in nzrange(A, j)
            nzv[k] *= d[rv[k]] * dj
        end
    end
end
