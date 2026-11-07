# Gather the masked elements into a dense array on the device, apply the inner term and scatter them back.
# A view with host indices would fall back to scalar indexing.
function RegularizedLeastSquares.prox!(reg::MaskedRegularization, x::AbstractGPUArray, args...)
  idx = findall(reg.mask)
  isempty(idx) && return x
  z = x[idx]
  prox!(reg.reg, z, args...)
  x[idx] = z
  return x
end
RegularizedLeastSquares.norm(reg::MaskedRegularization, x::AbstractGPUArray, args...) = norm(reg.reg, x[findall(reg.mask)], args...)
