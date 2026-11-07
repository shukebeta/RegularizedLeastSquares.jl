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

# A new state of the same type whose numeric arrays are replaced by `g(array, fieldname)`; scalars and counters are
# kept as they are. Shared by Adapt and element-type promotion so that both cover the same fields.
mapStateArrays(g, state::S) where S <: ProximalState = Base.typename(S).wrapper(map(fieldnames(S)) do f
  v = getfield(state, f)
  v isa AbstractArray{<:Number} ? g(v, f) : v
end...)

# Vectors are moved (or copied) and checked.
adaptState(ctx::CGNRAdaptContext, state::ProximalState) = mapStateArrays((v, f) -> adaptVector(ctx, v, "state.$f"), state)

# Backends: FISTA, OptISTA and POGM never change backends implicitly

proximalBackendError(solver, msg) = throw(ArgumentError("cannot initialize $(nameof(typeof(solver))): $msg"))

# Checks that the operators, every state buffer, `b` and an array-valued `x0` share one backend, using CGNR's backend
# tags. It runs before `init!` modifies the solver, so a failed call leaves buffers, progress, regularization terms,
# frame layout and `solver.state` unchanged.
function checkProximalBackend(solver::ProximalSolver, state, b, x0)
  name = nameof(typeof(solver))
  backend = nothing
  for s in leafStates(state), f in fieldnames(typeof(s))
    v = getfield(s, f)
    v isa AbstractArray || continue
    found = cgnrBackend(v)
    if isnothing(backend)
      backend = found
    elseif found != backend
      proximalBackendError(solver, "the solver state mixes backends $backend and $found; construct the solver on one " *
        "backend and move it with `Adapt.adapt(target, solver)`")
    end
  end
  isnothing(backend) && proximalBackendError(solver, "the solver state $(typeof(state)) contains no state vectors")
  for (opname, op) in (("A", solver.A), ("AHA", solver.AHA))
    found = cgnrOperatorBackend(op)
    isnothing(found) || found == backend || proximalBackendError(solver, "operator $opname is on backend $found but the " *
      "solver state is on backend $backend. Construct the solver on one backend and move it with `Adapt.adapt(target, solver)`")
  end
  for (what, v) in (("b", b), ("x0", x0))
    v isa AbstractArray || continue
    found = cgnrBackend(v)
    found == backend || proximalBackendError(solver, "$what is on backend $found but the solver state is on backend " *
      "$backend. $name does not change backends implicitly; move the solver with `Adapt.adapt(target, solver)` or pass " *
      "$what on the solver's backend")
  end
  return nothing
end

# A copy of `state` whose vectors have element type `T` and live on the state's backend, for a `b` of another element
# type (e.g. a complex `b` for a real solver). Scalars and counters are kept; `init!` resets them. Element types that
# the state type does not allow, e.g. `Float64` vectors with `Float32` scalars, throw here, before the solver changes.
promoteState(state::ProximalState, ::Type{T}) where T = mapStateArrays((v, _) -> similar(v, T, size(v)), state)
matchingState(state, b) = eltype(state.x) == eltype(b) ? state : promoteState(state, eltype(b))

# `b` with another element type than the state: continue with a promoted state on the solver's backend. Arrays with the
# state's element type, including views, dispatch to the algorithms' own methods and use the existing buffers.
# Each algorithm forwards its fallback `init!` method for vectors here.
function initPromoted!(solver::ProximalSolver, state::ProximalState, b::AbstractVector; kwargs...)
  checkProximalBackend(solver, state, b, get(kwargs, :x0, 0))
  state = promoteState(state, eltype(b))
  solver.state = state
  init!(solver, state, b; kwargs...)
end

# Frame transitions replace `solver.state`, so they are checked, and states of another element type are promoted,
# before anything changes. A matrix `b` copies the (promoted) current state once per column.
function init!(solver::ProximalSolver, state::AbstractSolverState, b::AbstractMatrix; x0 = 0, kwargs...)
  checkProximalBackend(solver, state, b, x0)
  template = matchingState(first(leafStates(state)), b)
  invoke(init!, Tuple{AbstractLinearSolver, AbstractSolverState, AbstractMatrix}, solver, template, b; x0, kwargs...)
end
# A vector `b` after a matrix continues with the state of the first column.
function init!(solver::ProximalSolver, state::AbstractMatrixSolverState, b::AbstractVector; x0 = 0, kwargs...)
  checkProximalBackend(solver, state, b, x0)
  single = matchingState(first(state.states), b)
  solver.state = single
  init!(solver, single, b; x0, kwargs...)
end
