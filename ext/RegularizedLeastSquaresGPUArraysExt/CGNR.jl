# GPU arrays are on the backend of their array type, e.g. `JLArray`; see `cgnrBackend` in `src/CGNR.jl`
RegularizedLeastSquares.cgnrBackend(S::Type{<:AbstractGPUArray}) = Base.typename(S).wrapper
