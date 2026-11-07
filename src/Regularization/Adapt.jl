########################################
# Adapt: rebuild built-in regularization terms on another array backend
########################################

const AdaptableRegularization = Union{NormalizedRegularization, FixedScaledRegularization, FixedParameterRegularization,
  AutoScaledRegularization, MaskedRegularization, TransformedRegularization, L2Regularization{<:AbstractArray}, TVRegularization}

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
Arrays and operators of the adapted term never share memory with `reg`; a same-backend target such as `Array`
returns independent copies. Masks of `MaskedRegularization` and the `dims` of `TVRegularization` are host metadata
and are copied, not moved.

Supported are the `L1`, `L2` (including array-valued weights), `L21`, `Nuclear`, `LLR`, `Real` and `Positive` terms,
`TVRegularization`, and the `Normalized`, `FixedScaled`, `FixedParameter`, `AutoScaled`, `Masked` and `Transformed`
wrappers with dense or LinearOperatorCollection.jl transforms that support Adapt.jl. Immutable terms holding only
scalars are backend-neutral and are returned as is. Within a supported term, other terms (such as
`PlugAndPlayRegularization`, `ProjectionRegularization` or custom terms) and transforms without an Adapt rule,
e.g. closure-based `LinearOperator`s, raise an `ArgumentError`, as do targets that do not move arrays or change
element types. Adapting such terms on their own is not supported yet: Adapt.jl's fallback may return them unchanged.
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
adaptRegularization(ctx::CGNRAdaptContext, reg) = adaptError(ctx, "unsupported regularization term $(typeof(reg))")

# `dims` is host metadata: ranges, integers and tuples are immutable, other arrays (including views) are copied
adaptDims(dims::AbstractArray) = dims isa AbstractRange ? dims : collect(dims)
adaptDims(dims) = dims

# Scale factors are scalars or `nothing`, except for array-valued L2 weights, where they can be arrays as well
adaptFactor(ctx::CGNRAdaptContext, factor::AbstractArray{<:Number}) = adaptVector(ctx, factor, "scale factor")
adaptFactor(ctx::CGNRAdaptContext, factor) = factor
