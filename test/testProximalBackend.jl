const PRLS = RegularizedLeastSquares

# CPU runs check a JLArray-backed solver, GPU runs their own arrayType.
proxTarget = arrayType === Array ? JLArray : arrayType
proxTag(x) = PRLS.cgnrBackend(x)
proxClose(x, ref) = norm(Array(x) - Array(ref)) <= 2e-4 * norm(Array(ref))

function proxFixture(T; m = 12, n = 6, k = 3, seed = 5959)
  rng = MersenneTwister(seed)
  A = T.(randn(rng, T, m, n) .+ 2 * vcat(Matrix(I, n, n), zeros(m - n, n)))
  return A, randn(rng, T, m, k), randn(rng, T, n)
end
# A fixed step size keeps solvers built from the same `A` comparable
proxRho(A) = real(eltype(A))(0.95 / opnorm(Array(A))^2)
proxSolver(S, A; rho = proxRho(A), kwargs...) = S(A; reg = L1Regularization(typeof(rho)(1e-3)), rho, iterations = 8, kwargs...)
proxOn(conv, solver) = conv === identity ? solver : Adapt.adapt(conv, solver)

proxChildren(s) = PRLS.leafStates(s)
proxBuffers(s) = Tuple(getfield(s, f) for f in fieldnames(typeof(s)) if getfield(s, f) isa AbstractArray)
proxProgress(s) = Tuple(getfield(s, f) for f in fieldnames(typeof(s)) if !(getfield(s, f) isa AbstractArray))
# FISTA and POGM swap their vectors while iterating, so buffers are compared as a set
proxSameBuffers(s, buffers) = issetequal(map(objectid, proxBuffers(s)), map(objectid, buffers))
proxOnBackend(s, tag) = all(c -> all(v -> proxTag(v) == tag, proxBuffers(c)), proxChildren(s))

# Everything `init!` could modify: the state object, the frame layout, every buffer, the progress and the regularization
function proxSnapshot(solver)
  st = solver.state
  children = proxChildren(st)
  frames = st isa PRLS.ProximalState ? nothing : (st.states, copy(st.states), copy(st.active))
  buffers = [map(v -> (v, copy(Array(v))), proxBuffers(c)) for c in children]
  return (; state = st, frames, children = copy(children), buffers, progress = map(proxProgress, children),
            reg = solver.reg, λ = λ(solver.reg))
end
function proxUnchanged(solver, snap)
  st = solver.state
  st === snap.state || return false
  if !isnothing(snap.frames)
    states, children, active = snap.frames
    (st.states === states && all(map(===, st.states, children)) && isequal(st.active, active)) || return false
  end
  for (c, c0, bufs, progress) in zip(proxChildren(st), snap.children, snap.buffers, snap.progress)
    c === c0 || return false
    all(((v, (v0, val)),) -> v === v0 && isequal(Array(v), val), zip(proxBuffers(c), bufs)) || return false
    isequal(proxProgress(c), progress) || return false
  end
  return solver.reg === snap.reg && λ(solver.reg) == snap.λ
end

# The call throws an ArgumentError that names the algorithm and the mismatch and points to explicit adaptation, before any change
function proxRejected(f, solver, needles...)
  snap = proxSnapshot(solver)
  err = try f(); nothing catch e; e end
  @test err isa ArgumentError
  msg = err isa Exception ? sprint(showerror, err) : ""
  @test occursin("cannot initialize $(nameof(typeof(solver)))", msg)
  @test occursin("Adapt.adapt(target, solver)", msg)
  for needle in needles
    @test occursin(string(needle), msg)
  end
  @test proxUnchanged(solver, snap)
end

# A solver on `conv`'s backend, fresh, initialized with a vector, or initialized as frames of a scheduler
function proxPrepared(S, A, B, conv, kind)
  solver = proxOn(conv, proxSolver(S, A))
  if kind === :vector
    init!(solver, conv(B[:, 1]))
    iterate(solver)
  elseif kind isa Type
    init!(solver, conv(B); scheduler = kind)
    solver.state.active[2] = false
    iterate(solver)
  end
  return solver
end

function testProximalBackendAccepts(S, target)
  for T in (Float32, ComplexF32)
    A, B, _ = proxFixture(T)
    b = B[:, 1]
    xref = solve!(proxSolver(S, A), b)
    for conv in (identity, target)
      solver = proxOn(conv, proxSolver(S, A))
      state = solver.state
      buffers = proxBuffers(state)
      padded = conv(vcat(b, zeros(T, 3)))
      # a view has another concrete type than the state vectors, but the same backend and element type
      for rhs in (conv(b), view(padded, 1:length(b)))
        init!(solver, rhs)
        @test solver.state === state && proxSameBuffers(state, buffers)
        @test all(iszero, Array(state.x))   # default x0 = 0
        @test iterate(solver) !== nothing
        @test proxClose(solve!(solver, rhs), xref)
        @test solver.state === state && proxSameBuffers(state, buffers)
      end
      # matrix views are split into columns on the solver's backend
      Bpad = conv(vcat(B, zeros(T, 3, size(B, 2))))
      X = solve!(solver, view(Bpad, 1:size(B, 1), :); scheduler = SequentialState)
      @test solver.state isa SequentialState && proxOnBackend(solver.state, proxTag(conv(b)))
      for j in 1:size(B, 2)
        @test proxClose(X[:, j], solve!(proxSolver(S, A), B[:, j]))
      end
    end
  end
end

# Working CPU element type conversions of a Float32 solver, with vectors and matrices
function testProximalNumeric(S)
  A, B, x0 = proxFixture(Float32)
  x0_64 = Float64.(x0)
  Bc = ComplexF32.(B .+ im .* reverse(B; dims = 1))
  reference(rhs) = solve!(proxSolver(S, A), rhs; x0)
  complexReference(rhs) = solve!(proxSolver(S, ComplexF32.(A)), rhs; x0 = ComplexF32.(x0))

  # Float64 x0 for Float32 b keeps the Float32 buffers
  solver = proxSolver(S, A)
  state = solver.state
  init!(solver, B[:, 1]; x0 = x0_64)
  @test solver.state === state && eltype(state.x) == Float32 && Array(state.x) == x0
  @test proxClose(solve!(solver, B[:, 1]; x0 = x0_64), reference(B[:, 1]))
  @test solver.state === state && eltype(state.x) == Float32
  for scheduler in (SequentialState, MultiThreadingState)
    solver = proxSolver(S, A)
    init!(solver, B; x0 = x0_64, scheduler)
    @test solver.state isa scheduler && length(solver.state.states) == size(B, 2)
    @test all(c -> eltype(c.x) == Float32 && Array(c.x) == x0, solver.state.states)
    X = solve!(solver, B; x0 = x0_64, scheduler)
    @test solver.state isa scheduler && all(c -> eltype(c.x) == Float32, solver.state.states)
    for j in 1:size(B, 2)
      @test proxClose(X[:, j], reference(B[:, j]))
    end
  end

  # ComplexF32 b promotes the buffers of a real solver to ComplexF32, also as frames
  solver = proxSolver(S, A)
  x = solve!(solver, Bc[:, 1]; x0)
  @test eltype(solver.state.x) == ComplexF32 && proxClose(x, complexReference(Bc[:, 1]))
  for scheduler in (SequentialState, MultiThreadingState)
    solver = proxSolver(S, A)
    init!(solver, B[:, 1])
    X = solve!(solver, Bc; x0, scheduler)
    @test solver.state isa scheduler && length(solver.state.states) == size(B, 2)
    @test all(c -> eltype(c.x) == ComplexF32, solver.state.states)
    for j in 1:size(B, 2)
      @test proxClose(X[:, j], complexReference(Bc[:, j]))
    end
    # matrix -> vector continues with the first column's ComplexF32 state
    first_state = first(solver.state.states)
    @test proxClose(solve!(solver, Bc[:, 2]; x0), complexReference(Bc[:, 2]))
    @test solver.state === first_state
  end

  # Float64 b for a Float32 solver stays unsupported: it throws and does not change the solver's precision
  for kind in (:fresh, :vector, SequentialState, MultiThreadingState)
    solver = proxPrepared(S, A, B, identity, kind)
    for rhs in (Float64.(B[:, 1]), Float64.(B))
      snap = proxSnapshot(solver)
      @test_throws Exception init!(solver, rhs)
      @test proxUnchanged(solver, snap)
    end
  end
end

# A state of `S` whose vectors are typed abstractly, so that they can live on different backends
function proxMixedState(s::ST, x) where ST
  T = eltype(s.x)
  return Base.typename(ST).wrapper{real(T), AbstractVector{T}}((f === :x ? x : getfield(s, f) for f in fieldnames(ST))...)
end

# Dense vector types outside `Array` and GPUArrays: one whose memory host BLAS cannot address, as Reactant's concrete
# arrays, and a host one that provides a pointer, as `SharedArray`
struct ProxForeignVector{T} <: DenseVector{T}
  data::Vector{T}
end
struct ProxHostVector{T} <: DenseVector{T}
  data::Vector{T}
end
for V in (ProxForeignVector, ProxHostVector)
  @eval Base.size(v::$V) = size(v.data)
  @eval Base.getindex(v::$V, i::Int) = v.data[i]
  @eval Base.setindex!(v::$V, x, i::Int) = (v.data[i] = x)
end
Base.unsafe_convert(::Type{Ptr{T}}, v::ProxHostVector{T}) where T = Base.unsafe_convert(Ptr{T}, v.data)

function testProximalBackendRejects(S, target)
  A, B, x0 = proxFixture(ComplexF32)
  b = B[:, 1]
  name = nameof(S)
  cpu, dev = proxTag(zeros(ComplexF32, 1)), proxTag(target(zeros(ComplexF32, 1)))
  @test dev != cpu && proxTag(view(target(b), 2:4)) == dev && proxTag(view(b, 2:4)) == cpu
  for (own, other, ownTag, otherTag) in ((identity, target, cpu, dev), (target, Array, dev, cpu))
    for kind in (:fresh, :vector, SequentialState, MultiThreadingState)
      solver = proxPrepared(S, A, B, own, kind)
      @test proxOnBackend(solver.state, ownTag)
      # vector b: same layout for a vector state, matrix -> vector transition for frames
      proxRejected(() -> init!(solver, other(b)), solver, "b is on backend $otherTag", ownTag)
      proxRejected(() -> solve!(solver, other(b)), solver, "b is on backend $otherTag", ownTag)
      proxRejected(() -> init!(solver, own(b); x0 = other(x0)), solver, "x0 is on backend $otherTag", ownTag)
      # matrix b: vector -> matrix transition for a vector state, new frames for frames
      for scheduler in (SequentialState, MultiThreadingState)
        proxRejected(() -> init!(solver, other(B); scheduler), solver, "b is on backend $otherTag", ownTag)
        proxRejected(() -> init!(solver, own(B); x0 = other(x0), scheduler), solver, "x0 is on backend $otherTag", ownTag)
      end
    end
  end

  # dense arrays outside GPUArrays without host memory are rejected by CPU solvers; host dense arrays are accepted
  # (`testProximalHostDense`), and CGNR's tags are unchanged
  solver = proxSolver(S, A)
  @test PRLS.proximalBackend(ProxForeignVector(b)) == ProxForeignVector && PRLS.cgnrBackend(ProxForeignVector(b)) == cpu
  proxRejected(() -> init!(solver, ProxForeignVector(copy(b))), solver, "b is on backend ProxForeignVector", cpu)
  proxRejected(() -> solve!(solver, ProxForeignVector(copy(b))), solver, "b is on backend ProxForeignVector", cpu)
  proxRejected(() -> init!(solver, b; x0 = ProxForeignVector(copy(x0))), solver, "x0 is on backend ProxForeignVector", cpu)

  # operators on another backend than the (default CPU) state, even if b matches the state: the implicit switch of
  # a solver constructed from GPU operators is no longer taken
  for (opname, solver) in (("A", proxSolver(S, target(A))),
                           ("AHA", proxSolver(S, nothing; AHA = target(A' * A), rho = proxRho(A))))
    for rhs in (b, target(b))
      proxRejected(() -> init!(solver, rhs), solver, "operator $opname is on backend $dev", cpu)
      proxRejected(() -> solve!(solver, rhs), solver, "operator $opname is on backend $dev", cpu)
    end
    proxRejected(() -> init!(solver, B), solver, "operator $opname is on backend $dev", cpu)
    proxRejected(() -> init!(solver, target(B)), solver, "operator $opname is on backend $dev", cpu)
  end

  # state vectors on different backends, inside one state or across frames
  solver = Adapt.adapt(target, proxSolver(S, A))
  devState = solver.state
  solver.state = proxMixedState(devState, Array(devState.x))
  proxRejected(() -> init!(solver, target(b)), solver, "mixes backends", name)
  proxRejected(() -> init!(solver, target(B)), solver, "mixes backends", name)
  cpuState = proxSolver(S, A).state
  solver.state = SequentialState(PRLS.AbstractSolverState{S}[devState, cpuState])
  proxRejected(() -> init!(solver, target(b)), solver, "mixes backends", name)
  proxRejected(() -> init!(solver, target(B)), solver, "mixes backends", name)
end

# A host dense array type other than `Array` works as `b` and `x0` like a `Vector`, for the family and for CGNR
function testProximalHostDense(S)
  A, B, x0 = proxFixture(ComplexF32)
  b = B[:, 1]
  @test PRLS.proximalBackend(ProxHostVector(b)) == PRLS.cgnrBackend(ProxHostVector(b)) == :CPU
  xref = solve!(proxSolver(S, A), b; x0)
  @test proxClose(solve!(proxSolver(S, A), ProxHostVector(copy(b)); x0 = ProxHostVector(copy(x0))), xref)
  cgnr() = CGNR(A; reg = L2Regularization(0.1f0), iterations = 4)
  @test proxClose(solve!(cgnr(), ProxHostVector(copy(b))), solve!(cgnr(), b))
end

# Explicitly adapted solvers keep their frame layout: vector -> matrix -> vector, compared with CPU solvers
function testProximalBackendFrames(S, target, scheduler)
  for T in (Float32, ComplexF32), conv in (Array, target)
    A, B, x0 = proxFixture(T)
    tag = proxTag(conv(zeros(T, 1)))
    solver = Adapt.adapt(conv, proxSolver(S, A))
    B_d, x0_d = conv(B), conv(x0)
    column(j) = (r = proxSolver(S, A); x = solve!(r, B[:, j]; x0); (x, r.state.iteration))

    # vector
    init!(solver, conv(B[:, 1]); x0 = x0_d)
    @test Array(solver.state.x) == x0 && solver.state.iteration == 0
    x, iters = column(1)
    @test proxClose(solve!(solver, conv(B[:, 1]); x0 = x0_d), x) && solver.state.iteration == iters

    # vector -> matrix: one state per column on the solver's backend, reset and iterated like the CPU columns
    init!(solver, B_d; x0 = x0_d, scheduler)
    @test solver.state isa scheduler && length(solver.state.states) == size(B, 2) && proxOnBackend(solver.state, tag)
    @test all(c -> Array(c.x) == x0 && c.iteration == 0, solver.state.states)
    X = solve!(solver, B_d; x0 = x0_d, scheduler)
    @test solver.state isa scheduler && length(solver.state.states) == size(B, 2) && proxOnBackend(solver.state, tag)
    for j in 1:size(B, 2)
      x, iters = column(j)
      @test proxClose(X[:, j], x) && solver.state.states[j].iteration == iters
    end
    @test Array(B_d) == B

    # matrix -> vector: the first column's state and buffers are reused on the solver's backend
    first_state = first(solver.state.states)
    buffers = proxBuffers(first_state)
    x, iters = column(2)
    @test proxClose(solve!(solver, conv(B[:, 2]); x0 = x0_d), x)
    @test solver.state === first_state && proxSameBuffers(first_state, buffers) && proxOnBackend(solver.state, tag)
    @test solver.state.iteration == iters
    @test Array(B_d) == B
  end
end

@testset "FISTA, OptISTA and POGM backends" begin
  @testset "$S" for S in (FISTA, OptISTA, POGM)
    @testset "$proxTarget" begin
      @testset "accepts" begin testProximalBackendAccepts(S, proxTarget) end
      @testset "rejects" begin testProximalBackendRejects(S, proxTarget) end
      for scheduler in (SequentialState, MultiThreadingState)
        @testset "frames $scheduler" begin testProximalBackendFrames(S, proxTarget, scheduler) end
      end
    end
    if arrayType === Array
      @testset "element types" begin testProximalNumeric(S) end
      @testset "host dense arrays" begin testProximalHostDense(S) end
    end
  end
end
