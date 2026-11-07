# Backend tags for CGNR's backend checks: the KernelAbstractions backend of a GPU array, e.g. `JLBackend`
# for `JLArray`. Sparse GPU arrays are tagged by the arrays that hold their entries, see `src/CGNR.jl`.
RegularizedLeastSquares.cgnrArrayBackend(x::AbstractGPUArray) = Base.typename(typeof(KernelAbstractions.get_backend(x))).wrapper

# Storage types are identified by the backend of an empty array of the same array type.
function RegularizedLeastSquares.cgnrBackend(S::Type{<:AbstractGPUArray})
  W = Base.typename(S).wrapper
  probe = try
    W{UInt8}(undef, 0)
  catch e
    throw(ArgumentError("cannot initialize CGNR: could not identify the backend of storage type $S: $(sprint(showerror, e))"))
  end
  return RegularizedLeastSquares.cgnrArrayBackend(probe)
end
