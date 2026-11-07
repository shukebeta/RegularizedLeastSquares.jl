using SparseArrays

const RLS = RegularizedLeastSquares
const LOB = LinearOperatorCollection.LinearOperators

# CPU runs check a JLArray-backed solver, GPU runs their own arrayType.
backendTarget = arrayType === Array ? JLArray : arrayType
backendTag(x) = RLS.cgnrBackend(x)

function backendFixture(T; m = 12, n = 6, k = 3, seed = 2828)
  rng = MersenneTwister(seed)
  A = T.(randn(rng, T, m, n) .+ 2 * vcat(Matrix(I, n, n), zeros(m - n, n)))
  return A, randn(rng, T, m, k), randn(rng, T, n)
end
backendSolver(A) = CGNR(A; reg = L2Regularization(real(eltype(A))(0.1)), iterations = 6)
closeTo(x, ref) = norm(Array(x) - ref) <= 2e-4 * norm(ref)

cgnrChildren(s::RLS.CGNRState) = [s]
cgnrChildren(s) = s.states
cgnrBuffers(s::RLS.CGNRState) = (s.x, s.x₀, s.pl, s.vl)
cgnrProgress(s::RLS.CGNRState) = (s.αl, s.βl, s.ζl, s.iteration, s.relTol, s.z0)
onBackend(s, tag) = all(c -> all(v -> backendTag(v) == tag, cgnrBuffers(c)), cgnrChildren(s))

# Everything `init!` could modify: the state object, the frame layout, every buffer and the progress
function solverSnapshot(solver)
  st = solver.state
  children = cgnrChildren(st)
  frames = st isa RLS.CGNRState ? nothing : (st.states, copy(st.states), copy(st.active))
  buffers = [map(v -> (v, copy(Array(v))), cgnrBuffers(c)) for c in children]
  return (; state = st, frames, children = copy(children), buffers, progress = map(cgnrProgress, children),
            L2 = solver.L2, λ = λ(solver.L2))
end
function unchanged(solver, snap)
  st = solver.state
  st === snap.state || return false
  if !isnothing(snap.frames)
    states, children, active = snap.frames
    (st.states === states && all(map(===, st.states, children)) && isequal(st.active, active)) || return false
  end
  for (c, c0, bufs, progress) in zip(cgnrChildren(st), snap.children, snap.buffers, snap.progress)
    c === c0 || return false
    all(((v, (v0, val)),) -> v === v0 && isequal(Array(v), val), zip(cgnrBuffers(c), bufs)) || return false
    isequal(cgnrProgress(c), progress) || return false
  end
  return solver.L2 === snap.L2 && λ(solver.L2) == snap.λ
end

# The call throws an ArgumentError that names the mismatch and points to explicit adaptation, before any change
function checkRejected(f, solver, needles...)
  snap = solverSnapshot(solver)
  err = try f(); nothing catch e; e end
  @test err isa ArgumentError
  msg = err isa Exception ? sprint(showerror, err) : ""
  @test occursin("Adapt.adapt(target, solver)", msg)
  for needle in needles
    @test occursin(string(needle), msg)
  end
  @test unchanged(solver, snap)
end

# A solver on `conv`'s backend, fresh, initialized with a vector, or initialized as frames of a scheduler
function preparedSolver(A, B, conv, kind)
  solver = conv === identity ? backendSolver(A) : Adapt.adapt(conv, backendSolver(A))
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

function testCGNRBackendAccepts(target)
  for T in (Float32, ComplexF32)
    A, B, _ = backendFixture(T)
    b = B[:, 1]
    xref = solve!(backendSolver(A), b)
    for conv in (identity, target)
      solver = conv === identity ? backendSolver(A) : Adapt.adapt(target, backendSolver(A))
      state = solver.state
      buffers = cgnrBuffers(state)
      padded = conv(vcat(b, zeros(T, 3)))
      # a view has another concrete type than the state vectors, but the same backend
      for rhs in (conv(b), view(padded, 1:length(b)))
        init!(solver, rhs)
        @test solver.state === state && all(map(===, cgnrBuffers(state), buffers))
        @test all(iszero, Array(state.x))   # default x0 = 0
        @test iterate(solver) !== nothing
        @test closeTo(solve!(solver, rhs), xref)
        @test solver.state === state && all(map(===, cgnrBuffers(state), buffers))
      end
    end
    # host sparse data is on the host backend
    solver = backendSolver(A)
    state = solver.state
    @test closeTo(solve!(solver, sparsevec(b)), xref)
    @test solver.state === state
  end
end

function testCGNRBackendRejects(target)
  A, B, x0 = backendFixture(ComplexF32)
  b = B[:, 1]
  cpu, dev = backendTag(zeros(ComplexF32, 1)), backendTag(target(zeros(ComplexF32, 1)))
  @test dev != cpu && backendTag(view(target(b), 2:4)) == dev && backendTag(view(b, 2:4)) == cpu
  for (own, other, ownTag, otherTag) in ((identity, target, cpu, dev), (target, Array, dev, cpu))
    for kind in (:fresh, :vector, SequentialState, MultiThreadingState)
      solver = preparedSolver(A, B, own, kind)
      @test onBackend(solver.state, ownTag)
      # vector b: same layout for a vector state, matrix -> vector transition for frames
      checkRejected(() -> init!(solver, other(b)), solver, "b is on backend $otherTag", ownTag)
      checkRejected(() -> solve!(solver, other(b)), solver, "b is on backend $otherTag", ownTag)
      checkRejected(() -> init!(solver, own(b); x0 = other(x0)), solver, "x0 is on backend $otherTag", ownTag)
      # matrix b: vector -> matrix transition for a vector state, new frames for frames
      for scheduler in (SequentialState, MultiThreadingState)
        checkRejected(() -> init!(solver, other(B); scheduler), solver, "b is on backend $otherTag", ownTag)
        checkRejected(() -> init!(solver, own(B); x0 = other(x0), scheduler), solver, "x0 is on backend $otherTag", ownTag)
      end
    end
  end

  # operators on another backend than the (default CPU) state, even if b matches the state
  for (name, solver) in (("A", backendSolver(target(A))),
                         ("AHA", CGNR(nothing; AHA = target(A' * A), normalizeReg = NoNormalization(), iterations = 6)))
    for rhs in (b, target(b))
      checkRejected(() -> init!(solver, rhs), solver, "operator $name is on backend $dev", cpu)
    end
    checkRejected(() -> init!(solver, B), solver, "operator $name is on backend $dev", cpu)
  end
end

# Operator with the storage type `S`
struct StorageOp{T, S} <: LOB.AbstractLinearOperator{T}
  nrow::Int
  ncol::Int
end
LOB.storage_type(::StorageOp{T, S}) where {T, S} = S
Base.size(op::StorageOp) = (op.nrow, op.ncol)

# Device sparse arrays outside the GPUArrays hierarchy, laid out like CUDA.jl 5's CuSparseVector and CuSparseMatrixCSC
struct LegacySparseVector{Tv, Ti} <: SparseArrays.AbstractSparseVector{Tv, Ti}
  iPtr::JLArray{Ti, 1}
  nzVal::JLArray{Tv, 1}
  len::Int
end
LegacySparseVector(v::SparseVector) = LegacySparseVector(JLArray(v.nzind), JLArray(v.nzval), length(v))
Base.size(v::LegacySparseVector) = (v.len,)
struct LegacySparseMatrixCSC{Tv, Ti} <: SparseArrays.AbstractSparseMatrix{Tv, Ti}
  colPtr::JLArray{Ti, 1}
  rowVal::JLArray{Ti, 1}
  nzVal::JLArray{Tv, 1}
  dims::NTuple{2, Int}
end
LegacySparseMatrixCSC(A::SparseMatrixCSC) = LegacySparseMatrixCSC(JLArray(A.colptr), JLArray(A.rowval), JLArray(A.nzval), size(A))
Base.size(A::LegacySparseMatrixCSC) = A.dims

# Sparse arrays on the backend of `target`, built by `spvec`/`spmat`, with `S` as an operator storage type
function testCGNRBackendSparse(target, spvec, spmat, S)
  A, B, x0 = backendFixture(Float32)
  b = B[:, 1]
  cpu, dev = backendTag(zeros(Float32, 1)), backendTag(target(zeros(Float32, 1)))
  @test backendTag(spvec(sparsevec(b))) == backendTag(spmat(sparse(A))) == backendTag(S) == dev
  @test backendTag(sparsevec(b)) == backendTag(sparse(A)) == cpu

  # device sparse data or operators with a CPU state
  solver = backendSolver(A)
  checkRejected(() -> init!(solver, spvec(sparsevec(b))), solver, "b is on backend $dev", cpu)
  checkRejected(() -> init!(solver, spmat(sparse(B))), solver, "b is on backend $dev", cpu)
  for op in (spmat(sparse(A)), StorageOp{Float32, S}(size(A)...))
    solver = CGNR(op; AHA = A' * A, iterations = 6)
    for rhs in (b, target(b))
      checkRejected(() -> init!(solver, rhs), solver, "operator A is on backend $dev", cpu)
    end
    checkRejected(() -> init!(solver, B), solver, "operator A is on backend $dev", cpu)
  end

  # device sparse data matches a device solver; only the check is run, as not every backend has sparse mul!
  solver = Adapt.adapt(target, backendSolver(A))
  @test isnothing(RLS.checkCGNRBackend(solver, solver.state, spvec(sparsevec(b)), 0))
  @test isnothing(RLS.checkCGNRBackend(solver, solver.state, target(b), spvec(sparsevec(x0))))
end

function testCGNRBackendFrames(target, scheduler)
  for T in (Float32, ComplexF32)
    A, B, x0 = backendFixture(T)
    dev = backendTag(target(zeros(T, 1)))
    solver = Adapt.adapt(target, backendSolver(A))
    B_d, x0_d = target(B), target(x0)
    columnReference(j) = (r = backendSolver(A); init!(r, B[:, j]; x0); r)

    # vector
    init!(solver, target(B[:, 1]); x0 = x0_d)
    ref = columnReference(1)
    @test Array(solver.state.x) == x0 && closeTo(solver.state.x₀, ref.state.x₀)
    @test closeTo(solve!(solver, target(B[:, 1]); x0 = x0_d), solve!(ref, B[:, 1]; x0))

    # vector -> matrix: one state per column on the target, initialized and iterated like the CPU columns
    init!(solver, B_d; x0 = x0_d, scheduler)
    @test solver.state isa scheduler && length(solver.state.states) == size(B, 2) && onBackend(solver.state, dev)
    refs = [columnReference(j) for j in 1:size(B, 2)]
    for (child, r) in zip(solver.state.states, refs)
      @test Array(child.x) == x0 && closeTo(child.x₀, r.state.x₀)
    end
    iterate(solver)
    foreach(iterate, refs)
    for (child, r) in zip(solver.state.states, refs)
      @test closeTo(child.x, r.state.x)
    end
    X = solve!(solver, B_d; x0 = x0_d, scheduler)
    @test solver.state isa scheduler && onBackend(solver.state, dev)
    for j in 1:size(B, 2)
      @test closeTo(X[:, j], solve!(backendSolver(A), B[:, j]; x0))
    end
    @test Array(B_d) == B

    # matrix -> vector: the first column's state is reused on the target
    first_state = first(solver.state.states)
    @test closeTo(solve!(solver, target(B[:, 2]); x0 = x0_d), solve!(backendSolver(A), B[:, 2]; x0))
    @test solver.state === first_state && onBackend(solver.state, dev)
    @test Array(B_d) == B
  end
end

@testset "CGNR backends" begin
  @testset "$backendTarget" begin
    @testset "accepts" begin testCGNRBackendAccepts(backendTarget) end
    @testset "rejects" begin testCGNRBackendRejects(backendTarget) end
    for scheduler in (SequentialState, MultiThreadingState)
      @testset "frames $scheduler" begin testCGNRBackendFrames(backendTarget, scheduler) end
    end
  end
  if arrayType === Array
    @testset "sparse outside GPUArrays" begin
      testCGNRBackendSparse(JLArray, LegacySparseVector, LegacySparseMatrixCSC, LegacySparseVector{Float32, Int})
    end
    # GPUArrays sparse arrays need GPUArrays/JLArrays releases with sparse support
    if isdefined(JLArrays, :JLSparseVector)
      @testset "GPUArrays sparse" begin
        testCGNRBackendSparse(JLArray, JLSparseVector, JLSparseMatrixCSC, JLSparseVector{Float32, Int})
      end
    end
  end
  if isdefined(@__MODULE__, :CUDA) && arrayType === CUDA.CuArray
    @testset "CUDA sparse" begin
      CS = CUDA.CUSPARSE
      testCGNRBackendSparse(CUDA.CuArray, CS.CuSparseVector, CS.CuSparseMatrixCSC, CS.CuSparseVector{Float32, Int32})
    end
  end
end
