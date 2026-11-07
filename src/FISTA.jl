export FISTA

mutable struct FISTA{matA, matAHA, R, RN} <: AbstractProximalGradientSolver
  A::matA
  AHA::matAHA
  reg::R
  proj::Vector{RN}
  normalizeReg::AbstractRegularizationNormalization
  verbose::Bool
  restart::Symbol
  iterations::Int64
  state::AbstractSolverState{<:FISTA}
end

mutable struct FISTAState{rT <: Real, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}} <: AbstractSolverState{FISTA}
  x::vecT
  x₀::vecT
  xᵒˡᵈ::vecT
  res::vecT
  ρ::rT
  theta::rT
  thetaᵒˡᵈ::rT
  iteration::Int64
  relTol::rT
  norm_x₀::rT
  rel_res_norm::rT
end


"""
    FISTA(A; AHA=A'*A, reg=L1Regularization(zero(real(eltype(AHA)))), normalizeReg=NoNormalization(), iterations=50, verbose = false, rho = 0.95 / power_iterations(AHA), theta=1, relTol=eps(real(eltype(AHA))), restart = :none)
    FISTA( ; AHA=,     reg=L1Regularization(zero(real(eltype(AHA)))), normalizeReg=NoNormalization(), iterations=50, verbose = false, rho = 0.95 / power_iterations(AHA), theta=1, relTol=eps(real(eltype(AHA))), restart = :none)

creates a `FISTA` object for the forward operator `A` or normal operator `AHA`.

# Required Arguments
  * `A`                                                 - forward operator
  OR
  * `AHA`                                               - normal operator (as a keyword argument)

# Optional Keyword Arguments
* `AHA`                                               - normal operator is optional if `A` is supplied
* `precon`                                            - preconditionner for the internal CG algorithm
* `reg::AbstractParameterizedRegularization`          - regularization term; can also be a vector of regularization terms
* `normalizeReg::AbstractRegularizationNormalization` - regularization normalization scheme; options are `NoNormalization()`, `MeasurementBasedNormalization()`, `SystemMatrixBasedNormalization()`
* `rho::Real`                                         - step size for gradient step; the default is `0.95 / max_eigenvalue` as determined with power iterations.
* `theta::Real`                                       - parameter for predictor-corrector step
* `relTol::Real`                                      - tolerance for stopping criterion
* `iterations::Int`                                   - maximum number of iterations
* `restart::Symbol`                                   - `:none`, `:gradient` options for restarting
* `verbose::Bool`                                     - print residual in each iteration

For a LinearOperatorCollection.jl operator `A`, the default `AHA = A'*A` is an opaque product; pass
`AHA = normalOperator(A)` to move the solver to another array backend with [`Adapt.adapt`](@ref Adapt.adapt_structure(::Any, ::RegularizedLeastSquares.ProximalSolver)).

# Backends

`FISTA` does not change its backend implicitly. The constructor allocates its state on the CPU, so a solver built
from GPU operators, or used with GPU data, must first be moved with `Adapt.adapt(target, solver)`. The operators,
the state, `b` and an array-valued `x0` must share one backend, otherwise `init!` and `solve!` throw an
`ArgumentError` before modifying the solver. Views and other wrappers of arrays on the solver's backend use the
solver's existing buffers. A complex `b` of the solver's precision for a real solver, e.g. `ComplexF32` for `Float32`,
continues with complex state vectors on the solver's backend. A `b` of higher precision, e.g. `Float64` for a
`Float32` solver, is not supported and throws before the solver is modified.

A matrix `b` creates one state per column, copied from the current state on the solver's backend, and runs them
with the `scheduler` passed to `init!`/`solve!`; a later vector `b` reuses the state of the first column.

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
FISTA(; AHA, kwargs...) = FISTA(nothing; AHA = AHA, kwargs...)

function FISTA(A
             ; AHA = A'*A
             , reg = L1Regularization(zero(real(eltype(AHA))))
             , normalizeReg = NoNormalization()
             , iterations = 50
             , verbose = false
             , rho = 0.95 / power_iterations(AHA; verbose)
             , theta = 1
             , relTol = eps(real(eltype(AHA)))
             , restart = :none
             )

  T  = eltype(AHA)
  rT = real(T)

  x    = Vector{T}(undef,size(AHA,2))
  x₀   = similar(x)
  xᵒˡᵈ = similar(x)
  res  = similar(x)
  res[1] = Inf # avoid spurious convergence in first iterations

  # Prepare regularization terms
  reg = copy(isa(reg, AbstractVector) ? reg : [reg])
  indices = findsinks(AbstractProjectionRegularization, reg)
  other = [reg[i] for i in indices]
  deleteat!(reg, indices)
  if length(reg) != 1
    error("FISTA does not allow for more additional regularization terms, found $(length(reg))")
  end
  other = identity.(other)
  reg = normalize(FISTA, normalizeReg, reg, A, nothing)

  state = FISTAState(x, x₀, xᵒˡᵈ, res, rT(rho), rT(theta), rT(theta), 0, rT(relTol), one(rT), rT(Inf))

  return FISTA(A, AHA, reg[1], other, normalizeReg, verbose, restart, iterations, state)
end

# `b` with another element type than the state, see `initPromoted!`
init!(solver::FISTA, state::FISTAState{rT, vecT}, b::AbstractVector; kwargs...) where {rT, vecT} = initPromoted!(solver, state, b; kwargs...)

"""
    init!(it::FISTA, b; x0 = 0, theta = 1)

(re-) initializes the FISTA iterator
"""
function init!(solver::FISTA, state::FISTAState{rT, vecT}, b::AbstractVector{T}; x0 = 0, theta=1) where {rT, T, vecT <: AbstractVector{T}}
  checkProximalBackend(solver, state, b, x0)
  if solver.A === nothing
    state.x₀ .= b
  else
    mul!(state.x₀, adjoint(solver.A), b)
  end
  state.iteration = 0

  state.norm_x₀ = norm(state.x₀)

  state.x    .= x0
  state.xᵒˡᵈ .= 0 # makes no difference in 1st iteration what this is set to

  state.res[:] .= rT(Inf)
  state.theta = theta
  state.thetaᵒˡᵈ = theta
  state.rel_res_norm = rT(Inf)
  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, state.x₀)
end

solverconvergence(state::FISTAState) = (; :residual => norm(state.res))


"""
  iterate(it::FISTA, iteration::Int=0)

performs one fista iteration.
"""
function iterate(solver::FISTA, state::FISTAState)
  if done(solver, state) return nothing end

  # momentum / Nesterov step
  # this implementation mimics BART, saving memory by first swapping x and xᵒˡᵈ before calculating x + α * (x - xᵒˡᵈ)
  tmp = state.xᵒˡᵈ
  state.xᵒˡᵈ = state.x
  state.x = tmp # swap x and xᵒˡᵈ
  state.x .*= ((1 - state.thetaᵒˡᵈ)/state.theta) # here we calculate -α * xᵒˡᵈ, where xᵒˡᵈ is now stored in x
  state.x .+= ((state.thetaᵒˡᵈ-1)/state.theta + 1) .* (state.xᵒˡᵈ) # add (α+1)*x, where x is now stored in xᵒˡᵈ

  # calculate residuum and do gradient step
  # solver.x .-= solver.ρ .* (solver.AHA * solver.x .- solver.x₀)
  mul!(state.res, solver.AHA, state.x)
  state.res .-= state.x₀
  state.x .-= state.ρ .* state.res

  state.rel_res_norm = norm(state.res) / state.norm_x₀
  solver.verbose && println("Iteration $(state.iteration); rel. residual = $(state.rel_res_norm)")

  # the two lines below are equivalent to the ones above and non-allocating, but require the 5-argument mul! function to implemented for AHA, i.e. if AHA is LinearOperator, it requires LinearOperators.jl v2
  # mul!(solver.x, solver.AHA, solver.xᵒˡᵈ, -solver.ρ, 1)
  # solver.x .+= solver.ρ .* solver.x₀

  # proximal map
  prox!(solver.reg, state.x, state.ρ * λ(solver.reg))

  for proj in solver.proj
    prox!(proj, state.x)
  end

  # gradient restart conditions
  if solver.restart == :gradient
    if real(state.res ⋅ (state.x .- state.xᵒˡᵈ) ) > 0 #if momentum is at an obtuse angle to the negative gradient
      solver.verbose && println("Gradient restart at iter $(state.iteration)")
      state.theta = 1
    end
  end

  # predictor-corrector update
  state.thetaᵒˡᵈ = state.theta
  state.theta = (1 + sqrt(1 + 4 * state.thetaᵒˡᵈ^2)) / 2

  state.iteration += 1
  # return the residual-norm as item and iteration number as state
  return state.x, state
end

@inline converged(::FISTA, state::FISTAState) = (state.rel_res_norm < state.relTol)

@inline done(solver::FISTA, state::FISTAState) = converged(solver, state) || state.iteration>=solver.iterations