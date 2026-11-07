########################################
# Adapt: rebuild built-in regularization terms on another array backend
########################################

const AdaptableRegularization = Union{NormalizedRegularization, FixedScaledRegularization, FixedParameterRegularization,
  AutoScaledRegularization, MaskedRegularization, TransformedRegularization, L2Regularization{<:AbstractArray}, TVRegularization,
  ProjectionRegularization, PlugAndPlayRegularization}

"""
    Adapt.adapt(target, reg::AbstractRegularization)

Return a copy of the regularization term `reg` whose numeric data lives on the backend of `target`, e.g.
`Adapt.adapt(CuArray, reg)`. The source term is left untouched.

```julia
reg = AutoScaledRegularization(TransformedRegularization(L1Regularization(0.1f0), WeightingOp(rand(Float32, 1024))))
gpu_reg = Adapt.adapt(CuArray, reg)
prox!(gpu_reg, CuArray(randn(Float32, 1024)))
```

Nested terms are rebuilt as one graph: a transform that appears several times in `reg` is adapted into a single
target object, and effective state is kept rather than recomputed, i.e. the factors of scaled terms (including an
initialized `AutoScaledRegularization`) and the initialized buffers and cached gradient operator of a `TVRegularization`.
The numeric arrays and operators the terms themselves hold never share memory with `reg`; a same-backend target such
as `Array` returns independent copies of them. Masks of `MaskedRegularization`, the `dims` of `TVRegularization` and
the `shape` of `PlugAndPlayRegularization` are host metadata and are copied, not moved. Callables follow the rules below.

Supported are the `L1`, `L2` (including array-valued weights), `L21`, `Nuclear`, `LLR`, `Real` and `Positive` terms,
`TVRegularization`, `ProjectionRegularization`, `PlugAndPlayRegularization`, and the `Normalized`, `FixedScaled`,
`FixedParameter`, `AutoScaled`, `Masked` and `Transformed` wrappers with dense or LinearOperatorCollection.jl
transforms that support Adapt.jl. Immutable terms holding only scalars are backend-neutral and are returned as is.

The projection function and the model and `input_transform` factory of a `PlugAndPlayRegularization` are adapted
with the Adapt rules for their types, called once with `target` itself, so rules specific to an adaptor apply:
- A callable type that holds data needs its own `Adapt.adapt_structure` rule; that rule decides what it moves, copies
  or shares. With a same-backend target such as `Array`, Adapt.jl's array rules return the arrays themselves.
- Closures are rebuilt by Adapt.jl's rule for functions. A closure that only captures numbers and plain arrays (such
  as `Array` or `CuArray`) gets independent copies even for a same-backend target; other captured values, e.g. views
  or `Ref`s, keep Adapt.jl's semantics and may stay shared.
- The factory stays a factory: the transforms it creates are made from the data on the target. The built-in factories
  `IdentityTransform`, `MinMaxTransform` and `ZTransform` hold no data; transform objects such as a
  `ClampedScalingTransform` retained by a factory can be adapted on their own, e.g. by the factory's rule.
- Global variables used by a function are not adapted. Opaque closures, closures capturing a reassigned variable
  (which Adapt.jl cannot rebuild) and callables without an Adapt rule raise an `ArgumentError`.

```julia
struct Shrink{V}
  w::V
end
(s::Shrink)(x) = x .* reshape(s.w, size(x))
Adapt.adapt_structure(to, s::Shrink) = Shrink(Adapt.adapt(to, s.w))

reg = PlugAndPlayRegularization(0.5f0; model = Shrink(rand(Float32, 8, 8)), shape = [8, 8])
gpu_reg = Adapt.adapt(CuArray, reg)
```

Within a supported term, other terms (such as custom terms) and transforms without an Adapt rule, e.g. closure-based
`LinearOperator`s, raise an `ArgumentError`, as do targets that do not move arrays or change element types. Adapting
such terms on their own is not supported: Adapt.jl's fallback may return them unchanged.
"""
Adapt.adapt_structure(to, reg::AdaptableRegularization) = adaptRegularization(regularizationAdaptContext(to), reg)
Adapt.adapt_structure(to, params::TVParams) = adaptRegularization(regularizationAdaptContext(to), params)

function regularizationAdaptContext(to)
  fail(msg) = throw(ArgumentError("cannot adapt regularization: unsupported Adapt target $to" * msg))
  to isa Type && isabstracttype(to) && fail(" (abstract type)")
  probe = Float32[0]
  probe′ = Adapt.adapt(to, probe)
  (probe′ isa AbstractArray && (to isa Type ? probe′ isa to : probe′ !== probe)) ||
    fail(": Adapt.adapt returned $(typeof(probe′)) for an array of type $(typeof(probe))")
  return CGNRAdaptContext(CGNRFreshArrays(to), arrayFamily(probe′), IdDict{Any, Any}(), IdDict{Any, Any}(), "regularization")
end

# Immutable terms that only hold scalars and tuples are backend-neutral
const ScalarRegularization = Union{L1Regularization, L2Regularization{<:Number}, L21Regularization, NuclearRegularization,
  LLRRegularization, RealRegularization, PositiveRegularization}
adaptRegularization(ctx::CGNRAdaptContext, reg::ScalarRegularization) = reg
adaptRegularization(ctx::CGNRAdaptContext, reg::L2Regularization{<:AbstractArray}) = L2Regularization(adaptVector(ctx, reg.λ, "L2Regularization.λ"))
adaptRegularization(ctx::CGNRAdaptContext, reg::NormalizedRegularization) =
  NormalizedRegularization(adaptRegularization(ctx, reg.reg), adaptFactor(ctx, reg.factor))
adaptRegularization(ctx::CGNRAdaptContext, reg::FixedScaledRegularization) =
  FixedScaledRegularization(adaptRegularization(ctx, reg.reg), adaptFactor(ctx, reg.factor))
adaptRegularization(ctx::CGNRAdaptContext, reg::FixedParameterRegularization) = FixedParameterRegularization(adaptRegularization(ctx, reg.reg))
adaptRegularization(ctx::CGNRAdaptContext, reg::AutoScaledRegularization) = get!(ctx.memo, reg) do
  adapted = AutoScaledRegularization(adaptRegularization(ctx, reg.reg))
  adapted.factor = adaptFactor(ctx, reg.factor)
  adapted
end
adaptRegularization(ctx::CGNRAdaptContext, reg::MaskedRegularization) = MaskedRegularization(adaptRegularization(ctx, reg.reg), copy(reg.mask))
function adaptRegularization(ctx::CGNRAdaptContext, reg::TransformedRegularization)
  reg.trafo isa Union{AbstractArray{<:Number}, AbstractLinearOperator} ||
    adaptError(ctx, "unsupported regularization transform of type $(typeof(reg.trafo))")
  return TransformedRegularization(adaptRegularization(ctx, reg.reg), adaptComponent(ctx, reg.trafo, "regularization transform"))
end
adaptRegularization(ctx::CGNRAdaptContext, reg::TVRegularization) = get!(ctx.memo, reg) do
  params = isnothing(reg.params) ? nothing : adaptRegularization(ctx, reg.params)
  TVRegularization(reg.λ, adaptDims(reg.dims), reg.shape, reg.iterationsTV, params)
end
# The cached gradient operator is adapted as is, it can differ from the one the term's current shape would create
adaptRegularization(ctx::CGNRAdaptContext, p::TVParams) = get!(ctx.memo, p) do
  TVParams(adaptVector(ctx, p.pq, "TVParams.pq"), adaptVector(ctx, p.rs, "TVParams.rs"), adaptVector(ctx, p.pqOld, "TVParams.pqOld"),
           adaptVector(ctx, p.xTmp, "TVParams.xTmp"), adaptComponent(ctx, p.∇, "TVParams.∇"))
end
adaptRegularization(ctx::CGNRAdaptContext, reg::ProjectionRegularization) =
  ProjectionRegularization(adaptCallable(ctx, reg.projFunc, "ProjectionRegularization.projFunc"))
adaptRegularization(ctx::CGNRAdaptContext, reg::PlugAndPlayRegularization) =
  PlugAndPlayRegularization(reg.λ; model = adaptCallable(ctx, reg.model, "PlugAndPlayRegularization.model"), shape = copy(reg.shape),
    input_transform = adaptCallable(ctx, reg.input_transform, "PlugAndPlayRegularization.input_transform"), ignoreIm = reg.ignoreIm)
adaptRegularization(ctx::CGNRAdaptContext, reg) = adaptError(ctx, "unsupported regularization term $(typeof(reg))")

# Callables are adapted by the Adapt rule of their type, called with the user's adaptor `ctx.to.to`, so that rules specific to
# that adaptor apply at every level. Only a closure that captures nothing but numbers and plain arrays, which Adapt hands to
# `adapt_storage`, goes through the fresh-array wrapper: it forwards each of them to the user's adaptor and copies what stays.
const StatelessTransformFactory = Union{Type{IdentityTransform}, Type{MinMaxTransform}, Type{ZTransform}}
adaptCallable(ctx::CGNRAdaptContext, f::StatelessTransformFactory, name) = f
adaptCallable(ctx::CGNRAdaptContext, f, name) = get!(ctx.memo, f) do
  to, F = ctx.to.to, Core.Typeof(f)
  rule = adaptRule(Adapt.adapt_structure, to, F)
  missingRule() = adaptError(ctx, "$name: callable of type $F has no Adapt rule, define `Adapt.adapt_structure` for it")
  if rule === fallbackRule(Adapt.adapt_structure, Core.OpaqueClosure)
    adaptError(ctx, "$name: opaque closure of type $F cannot be rebuilt")
  elseif rule === fallbackRule(Adapt.adapt_structure, Any)   # generic fallback, unless there is a storage rule
    adaptRule(Adapt.adapt_storage, to, F) === fallbackRule(Adapt.adapt_storage, Any) && missingRule()
  elseif rule === fallbackRule(Adapt.adapt_structure, Function)   # Adapt's closure rule
    startswith(String(nameof(F)), '#') || missingRule()   # compiler-generated function types only, not user `<: Function` structs
    plain(T) = T <: Union{Number, AbstractArray{<:Number}} && adaptRule(Adapt.adapt_structure, ctx.to, T) === fallbackRule(Adapt.adapt_structure, Any)
    f′ = Adapt.adapt(all(plain, fieldtypes(F)) ? ctx.to : to, f)
    f′ === f && Core.Box in fieldtypes(F) &&
      adaptError(ctx, "$name: closure of type $F captures a reassigned variable (Core.Box), which Adapt cannot rebuild")
    return f′
  end
  return Adapt.adapt(to, f)
end
adaptRule(fn, to, T::Type) = which(fn, Tuple{Core.Typeof(to), T})   # the method Adapt selects for adaptor `to` and a value of type T
fallbackRule(fn, T::Type) = which(fn, Tuple{Any, T})                 # Adapt's own method for any adaptor

# `dims` is host metadata: ranges, integers and tuples are immutable, other arrays (including views) are copied
adaptDims(dims::AbstractArray) = dims isa AbstractRange ? dims : collect(dims)
adaptDims(dims) = dims

# Scale factors are scalars or `nothing`, except for array-valued L2 weights, where they can be arrays as well
adaptFactor(ctx::CGNRAdaptContext, factor::AbstractArray{<:Number}) = adaptVector(ctx, factor, "scale factor")
adaptFactor(ctx::CGNRAdaptContext, factor) = factor

# Transform objects created by the input-transform factories, e.g. retained by a custom factory. Their arrays are adapted
# with the user's adaptor, so that its rules apply also below views; what still shares the source memory is copied.
const DataTransform = Union{MinMaxTransform, ZTransform, ClampedScalingTransform}
function Adapt.adapt_structure(to, t::T) where T <: DataTransform
  ctx = regularizationAdaptContext(to)
  return T(map(f -> adaptRetained(ctx, getfield(t, f), "$(nameof(T)).$f"), fieldnames(T))...)
end
function adaptRetained(ctx::CGNRAdaptContext, v::AbstractArray{<:Number}, name)
  v′ = Adapt.adapt(ctx.to.to, v)
  v′ isa AbstractArray && baseArray(v′) === baseArray(v) && (v′ = copy(v′))
  checkArray(ctx, v, v′, name)
  return v′
end
adaptRetained(ctx::CGNRAdaptContext, v, name) = v
