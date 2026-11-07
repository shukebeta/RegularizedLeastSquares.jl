# Backend tags for CGNR's backend checks: dense and sparse GPU arrays of one backend share its
# KernelAbstractions backend, e.g. `JLBackend` for `JLArray` and `JLSparseVector`.
# GPUArrays releases before its sparse array support have no `AbstractGPUSparseArray`.
@static if isdefined(GPUArrays, :AbstractGPUSparseArray)
  const GPUArrayTypes = Union{AbstractGPUArray, GPUArrays.AbstractGPUSparseArray}
  denseStorageType(S::Type) = S <: GPUArrays.AbstractGPUSparseArray ? GPUArrays.dense_array_type(S) : S
else
  const GPUArrayTypes = AbstractGPUArray
  denseStorageType(S::Type) = S
end

RegularizedLeastSquares.cgnrArrayBackend(x::GPUArrayTypes) = Base.typename(typeof(KernelAbstractions.get_backend(x))).wrapper

# Storage types are identified by the backend of an empty array of the corresponding dense type.
function RegularizedLeastSquares.cgnrBackend(S::Type{<:GPUArrayTypes})
  W = Base.typename(Base.unwrap_unionall(denseStorageType(S))).wrapper
  probe = try
    W{UInt8}(undef, 0)
  catch e
    throw(ArgumentError("cannot initialize CGNR: could not identify the backend of storage type $S: $(sprint(showerror, e))"))
  end
  return RegularizedLeastSquares.cgnrArrayBackend(probe)
end
