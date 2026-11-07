export L2Regularization

"""
    L2Regularization

Regularization term implementing the proximal map for Tikhonov regularization.
"""
struct L2Regularization{T} <: AbstractParameterizedRegularization{T}
  λ::T
  L2Regularization(λ::T; kargs...) where T = new{T}(λ)
end

"""
    prox!(reg::L2Regularization, x, λ)

performs the proximal map for Tikhonov regularization.
"""
function prox!(::L2Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T <: Real}
  x[:] .*= 1. ./ (1. .+ 2. .*λ)#*x
  return x
end

"""
    norm(reg::L2Regularization, x, λ)

returns the value of the L2-regularization term
"""
norm(::L2Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T <: Real} = λ*norm(x,2)^2
# Element i of x is weighted with λ[i] (linear indexing), as a reduction that also runs on GPU arrays
function norm(::L2Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::AbstractArray{T}) where {T <: Real}
  return mapreduce((w, xi) -> w * abs2(xi), +, view(vec(λ), 1:length(x)), vec(x); init = zero(real(eltype(x))))
end