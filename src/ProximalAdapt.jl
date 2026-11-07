const ProximalSolver = Union{FISTA, OptISTA, POGM}
const ProximalState = Union{FISTAState, OptISTAState, POGMState}

"""
    Adapt.adapt(target, solver::Union{FISTA, OptISTA, POGM})

returns a new solver of the same type whose operators, regularization terms, projections and state live on
`target` (e.g. `CuArray`, `ROCArray` or `JLArray`; `Array` gives a CPU copy). Initialized and partially
iterated solvers keep their effective state: step size, momentum and restart scalars, iteration count, normalized
regularization factors and warmed term buffers are copied as they are, not recomputed, so iterating the result
resumes where the source stopped. Frame states created by matrix-valued `init!`/`solve!` (`SequentialState`,
`MultiThreadingState`) keep their active flags. The source solver is not modified.

Supported operators are dense arrays and LinearOperatorCollection.jl operators with Adapt rules. An `AHA` of type
`NormalOp` whose parent is `A`, or the inner operand `M` of a weighted `ProdOp(W, M)` (as built by
`normalOperator(A)`), is rebuilt on the adapted `A`/`M`. Regularization terms and projections are rebuilt with the Adapt rules of the
regularization terms, including the ownership of callables.

The operators, state vectors and buffers of the built-in terms of the result share no data with the source. Data
owned by callables, such as a plug-and-play model, is moved by the callable's own Adapt rule, which decides whether it
is copied or shared; e.g. a rule that adapts its arrays with `Adapt.adapt` shares them with the source on a target
that leaves them in place, such as `Array`. An operator that is shared in the source, e.g. a factor of `A` that is
also a regularization transform, stays one shared object in the result; arrays shared by several operators may be
adapted into separate copies.

The default `AHA = A' * A` of a LinearOperatorCollection.jl operator `A` is an opaque product that cannot be
rebuilt: construct such a solver with `AHA = normalOperator(A)`, or with an `AHA` that has an Adapt rule, to make it
adaptable. Unlike `A' * A`, that `NormalOp`
supports `AHA * x` but not `adjoint(AHA) * x` or `transpose(AHA) * x`, which the solvers do not use.

An `ArgumentError` is thrown instead of returning a partially moved solver for unsupported targets (e.g. `Int`,
`AbstractArray` or an adaptor that leaves the state unchanged), element-type-changing targets, operators without an
Adapt rule (closure-based `LinearOperator`s and products such as `A' * A` of operators), and unsupported custom
states or regularization terms.
"""
function Adapt.adapt_structure(to, solver::S) where S <: ProximalSolver
  ctx = solverAdaptContext(to, solver, ProximalState)
  opaqueNormal(ctx, solver.A, solver.AHA) &&
    adaptError(ctx, "AHA of type $(typeof(solver.AHA)) has no Adapt rule, e.g. the default A' * A of a LinearOperatorCollection.jl " *
                    "operator A; construct the solver with AHA = normalOperator(A), or with an AHA that has an Adapt rule")
  A = adaptComponent(ctx, solver.A, "A")
  AHA = adaptNormal(ctx, solver.A, A, solver.AHA)
  reg = adaptRegularization(ctx, solver.reg)
  proj = isempty(solver.proj) ? copy(solver.proj) : identity.([adaptRegularization(ctx, p) for p in solver.proj])
  state = adaptState(ctx, solver.state)
  adapted = (; A, AHA, reg, proj, state)
  return Base.typename(S).wrapper((f in keys(adapted) ? adapted[f] : getfield(solver, f) for f in fieldnames(S))...)
end

# A closure-based AHA, e.g. the default A' * A, of an operator A that has a structured normal operator, whose Adapt
# fallback returns it unchanged. A LinearOperator with its own Adapt rule is rebuilt here once (the result is memoized
# for adaptNormal), and a failure of that rule is reported as such.
function opaqueNormal(ctx::CGNRAdaptContext, A, AHA)
  (A isa LinearOperatorCollection.AbstractLinearOperatorFromCollection && AHA isa LinearOperators.LinearOperator) || return false
  AHA′ = try
    Adapt.adapt(ctx.to, AHA)
  catch e
    e isa LinearOperators.LinearOperatorException || rethrow()
    adaptError(ctx, "AHA of type $(typeof(AHA)) could not be adapted: $(sprint(showerror, e))")
  end
  return AHA′ === AHA
end

# Vectors are moved (or copied) and checked; scalars and counters are kept as they are.
adaptState(ctx::CGNRAdaptContext, state::S) where S <: ProximalState = Base.typename(S).wrapper(map(fieldnames(S)) do f
  v = getfield(state, f)
  v isa AbstractArray{<:Number} ? adaptVector(ctx, v, "state.$f") : v
end...)
