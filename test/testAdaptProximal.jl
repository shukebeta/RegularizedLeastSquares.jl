# Public Adapt.adapt for FISTA, OptISTA and POGM.
# Reuses adaptTargets/NoopAdaptor/Float32ToFloat64Adaptor (testAdapt.jl) and RegModel/CustomRegTerm/regRejects
# (testAdaptRegularization.jl); scalar indexing stays disabled.

proxLeaves(s::Union{SequentialState, MultiThreadingState}) = s.states
proxLeaves(s) = [s]
proxFields(s) = map(f -> getfield(s, f), fieldnames(typeof(s)))
proxVectors(s) = filter(v -> v isa AbstractArray, proxFields(s))
proxScalars(s) = filter(v -> !(v isa AbstractArray), proxFields(s))
proxSnapshot(solver) = map(l -> (map(v -> copy(Array(v)), proxVectors(l)), proxScalars(l)), proxLeaves(solver.state))
proxSettings(solver) = map(f -> getfield(solver, f), filter(!in((:A, :AHA, :reg, :proj, :state)), fieldnames(typeof(solver))))
proxRestart(Solver) = Solver === OptISTA ? (;) : (; restart = :gradient)

struct CustomProxState{S} <: RegularizedLeastSquares.AbstractSolverState{S} end

# Square, well-conditioned systems (n = 6, a 2×3 image for TV and PnP); returns A, constructor keywords and the rhs
function proxSystem(T, graph; seed = 58)
  rng = MersenneTwister(seed)
  M = Matrix{T}(2I, 6, 6) .+ randn(rng, T, 6, 6) ./ 4
  w, b = rand(rng, Float32, 6) .+ 0.5f0, randn(rng, T, 6)
  graph == :dense && return M, (;), b
  graph == :normal && return nothing, (; AHA = M' * M), M' * b
  graph == :collection && (W = WeightingOp(T.(w)); return W, (; AHA = normalOperator(W)), b)
  P = ProdOp(WeightingOp(T.(w)), M)
  return P, (; AHA = normalOperator(P)), b
end

function proxTerms(T, terms)
  terms == :l1 && return [L1Regularization(0.01f0), T <: Real ? PositiveRegularization() : RealRegularization()], (;)
  terms == :scaled && return [L1Regularization(0.01f0)], (; normalizeReg = MeasurementBasedNormalization())
  terms == :tv && return [TVRegularization(0.01f0; shape = (2, 3), iterationsTV = 3)], (;)
  return [PlugAndPlayRegularization(0.2f0; model = RegModel(regWeights(6; seed = 61)), shape = [2, 3])], (;)
end

# the adapted solver lives on target, keeps settings and effective scalars, and shares no state vector with source
function checkProxMoved(moved, source, target)
  @test typeof(moved).name === typeof(source).name
  @test isequal(proxSettings(moved), proxSettings(source))
  @test typeof(moved.state).name === typeof(source.state).name
  for (m, s) in zip(proxLeaves(moved.state), proxLeaves(source.state))
    @test isequal(proxScalars(m), proxScalars(s))
    @test all(((v, w),) -> v isa target && v !== w && eltype(v) == eltype(w) && isequal(Array(v), Array(w)), zip(proxVectors(m), proxVectors(s)))
  end
  @test isnothing(source.A) ? isnothing(moved.A) : moved.A isa target || onTarget(moved.A, target)
end

function proxIterate!(solver, k)
  for _ in 1:k
    iterate(solver)
  end
  return solversolution(solver)
end

function testProxAdaptSolve(target, Solver, T, graph, terms)
  A, kw, b = proxSystem(T, graph)
  reg, rkw = proxTerms(T, terms)
  x0 = randn(MersenneTwister(59), T, 6) ./ 4
  extra = terms == :l1 ? (;) : proxRestart(Solver)
  source = Solver(A; kw..., rkw..., extra..., reg, iterations = 20)
  graph == :collection && @test (m = Adapt.adapt(target, source); m.AHA.parent === m.A)
  graph == :weighted && @test (m = Adapt.adapt(target, source); m.AHA.parent === m.A.B)

  # uninitialized: adapt -> init! -> iterate on the target
  moved = Adapt.adapt(target, source)
  checkProxMoved(moved, source, target)
  reference = deepcopy(source)
  init!(moved, target(b); x0 = target(x0)); init!(reference, b; x0)
  @test Array(proxIterate!(moved, 4)) ≈ proxIterate!(reference, 4) rtol = 2e-4

  # progressed: resume on the target from the source's effective state
  init!(source, b; x0)
  proxIterate!(source, 3)
  before = proxSnapshot(source)
  moved = Adapt.adapt(target, source)
  checkProxMoved(moved, source, target)
  terms == :scaled && @test moved.reg.factor == source.reg.factor
  reference = deepcopy(source)
  @test Array(proxIterate!(moved, 3)) ≈ proxIterate!(reference, 3) rtol = 2e-4
  @test isequal(proxSnapshot(source), before)

  # round trip to Array resumes as well; zeroing the copies leaves the source unchanged
  back = Adapt.adapt(Array, moved)
  checkProxMoved(back, moved, Array)
  @test proxIterate!(back, 2) ≈ Array(proxIterate!(moved, 2)) rtol = 2e-4
  for s in (moved, back), l in proxLeaves(s.state), v in proxVectors(l)
    fill!(v, zero(eltype(v)))
  end
  @test isequal(proxSnapshot(source), before)
  if graph in (:collection, :weighted)
    W, W′ = graph == :collection ? (source.A, moved.A) : (source.A.A, moved.A.A)
    keep = copy(W.weights)
    fill!(W′.weights, 0)
    @test W.weights == keep
  end
end

function testProxAdaptSharing(target, Solver)
  w(seed) = rand(MersenneTwister(seed), Float32, 6) .+ 0.5f0
  M = Matrix{Float32}(2I, 6, 6) .+ randn(MersenneTwister(62), Float32, 6, 6) ./ 4
  b = randn(MersenneTwister(63), Float32, 6)
  function resumes(source)
    init!(source, b)
    proxIterate!(source, 2)
    moved, reference = Adapt.adapt(target, source), deepcopy(source)
    @test Array(proxIterate!(moved, 3)) ≈ proxIterate!(reference, 3) rtol = 2e-4
    return moved
  end
  isolated(moved, src) = (keep = copy(src.weights); fill!(moved.weights, 0); src.weights == keep && moved.weights !== src.weights)

  # a factor of A used as a regularization transform
  P = ProdOp(WeightingOp(w(64)), M)
  source = Solver(P; AHA = normalOperator(P), reg = TransformedRegularization(L1Regularization(0.01f0), P.A))
  moved = resumes(source)
  @test moved.reg.trafo === moved.A.A && moved.AHA.parent === moved.A.B
  @test isolated(moved.A.A, P.A) && source.reg.trafo === P.A

  # a transform containing A
  W = WeightingOp(w(65))
  source = Solver(W; AHA = normalOperator(W), reg = TransformedRegularization(L1Regularization(0.01f0), ProdOp(W, WeightingOp(w(66)))))
  moved = resumes(source)
  @test moved.reg.trafo.A === moved.A && moved.AHA.parent === moved.A
  @test isolated(moved.A, W)

  # an operator repeated inside A
  Wr = WeightingOp(w(67))
  P = ProdOp(Wr, ProdOp(Wr, M))
  source = Solver(P; AHA = normalOperator(P), reg = L1Regularization(0.01f0))
  moved = resumes(source)
  # normalOperator(P) is a ProdNormalOp whose inner normal operator is built on the repeated factor
  @test moved.A.A === moved.A.B.A && moved.AHA.normalOpInner.parent === moved.A.A
  @test isolated(moved.A.A, Wr)

  # an operator shared by two terms
  S = WeightingOp(w(68))
  source = Solver(M; reg = [TransformedRegularization(L1Regularization(0.01f0), S), TransformedRegularization(PositiveRegularization(), S)])
  moved = resumes(source)
  @test moved.reg.trafo === moved.proj[1].trafo
  @test isolated(moved.reg.trafo, S)

  # the rebuilt AHA nested in a stored projection and in a term. Its NormalOp has no adjoint, so only OptISTA,
  # which stores but does not apply projections, can iterate the projection case.
  W = WeightingOp(w(69))
  H = normalOperator(W)
  inner() = ProdOp(H, WeightingOp(w(70)))
  source = Solver(W; AHA = H, reg = [L1Regularization(0.01f0), TransformedRegularization(PositiveRegularization(), inner())], rho = 0.1f0)
  moved = Solver === OptISTA ? resumes(source) : Adapt.adapt(target, source)
  @test moved.proj[1].trafo.A === moved.AHA && moved.AHA.parent === moved.A
  @test isolated(moved.A, W) && source.AHA.parent === W
  source = Solver(W; AHA = H, reg = TransformedRegularization(L1Regularization(0.01f0), inner()), rho = 0.1f0)
  moved = Adapt.adapt(target, source)
  @test moved.reg.trafo.A === moved.AHA && moved.AHA.parent === moved.A
end

function testProxAdaptDefaultNormal(target, Solver, T)
  w = T.(rand(MersenneTwister(71), Float32, 6) .+ 0.5f0)
  v = randn(MersenneTwister(72), T, 6)
  N = Matrix(Diagonal(w))' * Matrix(Diagonal(w))
  # the default stays A'*A, with its forward, adjoint and transpose action
  source = Solver(WeightingOp(w); reg = L1Regularization(0.01f0))
  @test !(source.AHA isa NormalOp)
  @test source.AHA * v ≈ N * v && adjoint(source.AHA) * v ≈ N' * v && transpose(source.AHA) * v ≈ transpose(N) * v
  x = source.state.x
  @test regRejects(() -> Adapt.adapt(target, source), "normalOperator")
  @test source.state.x === x
  # a dense A keeps its matrix default, which moves
  M = Matrix(Diagonal(w))
  moved = Adapt.adapt(target, Solver(M; reg = L1Regularization(0.01f0)))
  @test moved.AHA isa target && Array(moved.AHA) == M' * M
end

function testProxAdaptErrors(target, Solver)
  M = Matrix{Float32}(2I, 6, 6)
  source = Solver(M; reg = L1Regularization(0.01f0))
  x = source.state.x
  for bad in (Int, AbstractArray, NoopAdaptor(), target{Float64})
    @test_throws ArgumentError Adapt.adapt(bad, source)
  end
  target === Array && @test_throws ArgumentError Adapt.adapt(Float32ToFloat64Adaptor(), source)
  @test source.state.x === x
  @test_throws ArgumentError Adapt.adapt(target, Solver(LinearOperator(M); reg = L1Regularization(0.01f0)))
  L = LinearOperator(M)
  @test_throws ArgumentError Adapt.adapt(target, Solver(M; AHA = L' * L, reg = L1Regularization(0.01f0)))
  @test regRejects(() -> Adapt.adapt(target, Solver(M; reg = [L1Regularization(0.01f0), CustomRegTerm()])), "CustomRegTerm")
  custom = Solver(M; reg = L1Regularization(0.01f0))
  custom.state = CustomProxState{Solver}()
  @test regRejects(() -> Adapt.adapt(target, custom), "unsupported solver state")
  empty = Solver(M; reg = L1Regularization(0.01f0))
  empty.state = SequentialState([empty.state])
  empty!(empty.state.states)
  @test regRejects(() -> Adapt.adapt(target, empty), "contains no $(nameof(Solver)) states")
end

function testProxAdaptFrames(target, Solver, scheduler)
  A, _, _ = proxSystem(ComplexF32, :dense)
  rhs = randn(MersenneTwister(73), ComplexF32, 6, 3)
  source = Solver(A; reg = L1Regularization(0.01f0), iterations = 10)
  init!(source, rhs; scheduler)
  source.state.active[2] = false
  iterate(source)
  before = proxSnapshot(source)
  moved = Adapt.adapt(target, source)
  checkProxMoved(moved, source, target)
  @test moved.state.active == source.state.active && moved.state.active !== source.state.active
  reference = deepcopy(source)
  iterate(moved); iterate(reference)
  @test Array(solversolution(moved)) ≈ solversolution(reference) rtol = 2e-4
  @test isequal(proxSnapshot(source), before)
end

@testset "Proximal Adapt" begin
  for target in adaptTargets
    @testset "$target" begin
      for Solver in (FISTA, OptISTA, POGM)
        @testset "$Solver" begin
          for T in (Float32, ComplexF32)
            for graph in (:dense, :normal, :collection, :weighted)
              @testset "$T $graph" begin testProxAdaptSolve(target, Solver, T, graph, :l1) end
            end
            for terms in (:scaled, :tv, :pnp)
              @testset "$T $terms" begin testProxAdaptSolve(target, Solver, T, :dense, terms) end
            end
            @testset "default normal $T" begin testProxAdaptDefaultNormal(target, Solver, T) end
          end
          @testset "sharing" begin testProxAdaptSharing(target, Solver) end
          @testset "errors" begin testProxAdaptErrors(target, Solver) end
          for scheduler in (SequentialState, MultiThreadingState)
            @testset "frames $scheduler" begin testProxAdaptFrames(target, Solver, scheduler) end
          end
        end
      end
    end
  end
end
