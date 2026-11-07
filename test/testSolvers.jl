Random.seed!(12345)

function testRealLinearSolver(; arrayType=Array, elType=Float32)
    A = rand(elType, 3, 2)
    x = rand(elType, 2)
    b = A * x

    solvers = linearSolverListReal()

    @testset for solver in solvers
        @test try
            S = createLinearSolver(solver, arrayType(A), iterations=200)
            # CGNR does not switch backends implicitly
            solver <: CGNR && (S = Adapt.adapt(arrayType, S))
            x_approx = Array(solve!(S, arrayType(b)))
            @info "Testing solver $solver: $x ≈ $x_approx"
            @test x_approx ≈ x rtol = 0.1
            true
        catch e
            @error e
            false
        end skip = arrayType != Array && solver <: AbstractDirectSolver #
    end

    # CGNR built on the CPU and moved to adaptType with Adapt.adapt
    S = Adapt.adapt(adaptType, createLinearSolver(CGNR, A, iterations=200))
    @test Array(solve!(S, adaptType(b))) ≈ x rtol = 0.1
end

function testComplexLinearSolver(; arrayType=Array, elType=Float32)
    A = rand(elType, 3, 2) + im * rand(elType, 3, 2)
    x = rand(elType, 2) + im * rand(elType, 2)
    b = A * x

    solvers = linearSolverList()

    @testset for solver in solvers
        @test try
            S = createLinearSolver(solver, arrayType(A), iterations=100)
            # CGNR does not switch backends implicitly
            solver <: CGNR && (S = Adapt.adapt(arrayType, S))
            x_approx = Array(solve!(S, arrayType(b)))
            @info "Testing solver $solver: $x ≈ $x_approx"
            @test x_approx ≈ x rtol = 0.1
            true
        catch e
            @error e
            false
        end skip = arrayType != Array && solver <: AbstractDirectSolver
    end

    # CGNR built on the CPU and moved to adaptType with Adapt.adapt
    S = Adapt.adapt(adaptType, createLinearSolver(CGNR, A, iterations=100))
    @test Array(solve!(S, adaptType(b))) ≈ x rtol = 0.1
end

function testComplexLinearAHASolver(; arrayType=Array, elType=Float32)
    A = rand(elType, 3, 2) + im * rand(elType, 3, 2)
    x = rand(elType, 2) + im * rand(elType, 2)
    AHA = A' * A
    b = AHA * x

    solvers = filter(s -> s ∉ [DirectSolver, PseudoInverse, Kaczmarz], linearSolverListReal())

    @testset for solver in solvers
        @test try
            S = createLinearSolver(solver, nothing; AHA=arrayType(AHA), iterations=100)
            # CGNR does not switch backends implicitly
            solver <: CGNR && (S = Adapt.adapt(arrayType, S))
            x_approx = Array(solve!(S, arrayType(b)))
            @info "Testing solver $solver: $x ≈ $x_approx"
            @test x_approx ≈ x rtol = 0.1
            true
        catch e
            @error e
            false
        end
    end

    # CGNR built on the CPU and moved to adaptType with Adapt.adapt
    S = Adapt.adapt(adaptType, createLinearSolver(CGNR, nothing; AHA=AHA, iterations=100))
    @test Array(solve!(S, adaptType(b))) ≈ x rtol = 0.1
end

function testConvexLinearSolver(; arrayType=Array, elType=Float32)
    # fully sampled operator, image and data
    N = 256
    numPeaks = 5
    F = [1 / sqrt(N) * exp(-2im * π * j * k / N) for j = 0:N-1, k = 0:N-1]
    x = zeros(N)
    for i = 1:3
        x[rand(1:N)] = rand()
    end
    b = 1 / sqrt(N) * fft(x)

    # random undersampling
    idx = sort(unique(rand(1:N, div(N, 2))))
    b = arrayType(b[idx])
    F = arrayType(F[idx, :])

    for solver in [POGM, OptISTA, FISTA, ADMM]
        reg = L1Regularization(elType(1e-3))
        S = createLinearSolver(
            solver,
            F;
            reg=reg,
            iterations=200,
            normalizeReg=NoNormalization(),
        )
        x_approx = Array(solve!(S, b))
        @info "Testing solver $solver w/o restart: relative error = $(norm(x - x_approx) / norm(x))"
        @test x ≈ x_approx rtol = 0.1

        #additionally test the gradient restarting scheme
        if solver == POGM || solver == FISTA
            S = createLinearSolver(
                solver,
                F;
                reg=reg,
                iterations=200,
                normalizeReg=NoNormalization(),
                restart=:gradient,
            )
            x_approx = Array(solve!(S, b))
            @info "Testing solver $solver w/ gradient restart: relative error = $(norm(x - x_approx) / norm(x))"
            @test x ≈ x_approx rtol = 0.1
        end

        # test invariance to the maximum eigenvalue
        reg = L1Regularization(elType(reg.λ * length(b) / norm(b, 1)))
        scale_F = 1e3
        S = createLinearSolver(
            solver,
            F .* scale_F;
            reg=reg,
            iterations=200,
            normalizeReg=MeasurementBasedNormalization(),
        )
        x_approx = Array(solve!(S, b))
        x_approx .*= scale_F
        @info "Testing solver $solver w/o restart and after re-scaling: relative error = $(norm(x - x_approx) / norm(x))"
        @test x ≈ x_approx rtol = 0.1
    end

    # test ADMM with option vary_rho
    solver = ADMM
    reg = L1Regularization(elType(1.e-3))
    S = createLinearSolver(
        solver,
        F;
        reg=reg,
        iterations=200,
        normalizeReg=NoNormalization(),
        rho=1e6,
        vary_rho=:balance,
        verbose=false,
    )
    x_approx = Array(solve!(S, b))
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    S = createLinearSolver(
        solver,
        F;
        reg=reg,
        iterations=200,
        normalizeReg=NoNormalization(),
        rho=1e-6,
        vary_rho=:balance,
        verbose=false,
    )
    x_approx = Array(solve!(S, b))
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    # the PnP scheme only increases rho, hence we only test it with a small initial rho
    S = createLinearSolver(
        solver,
        F;
        reg=reg,
        iterations=200,
        normalizeReg=NoNormalization(),
        rho=1e-6,
        vary_rho=:PnP,
        verbose=false,
    )
    x_approx = Array(solve!(S, b))
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    ##
    solver = SplitBregman
    reg = L1Regularization(elType(2e-3))
    S = createLinearSolver(
        solver,
        F;
        reg=reg,
        iterations=5,
        iterationsInner=40,
        rho=1.0,
        normalizeReg=NoNormalization(),
    )
    x_approx = Array(solve!(S, b))
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    reg = L1Regularization(elType(reg.λ * length(b) / norm(b, 1)))
    S = createLinearSolver(
        solver,
        F;
        reg=reg,
        iterations=5,
        iterationsInner=40,
        rho=1.0,
        normalizeReg=MeasurementBasedNormalization(),
    )
    x_approx = Array(solve!(S, b))
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    #=
    solver = PrimalDualSolver
    reg = [L1Regularization(1.e-4), TVRegularization(1.e-4, shape = (0,0))]
    FR = [real.(F ./ norm(F)); imag.(F ./ norm(F))]
    bR = [real.(b ./ norm(F)); imag.(b ./ norm(F))]
    S = createLinearSolver(
        solver,
        FR;
        reg = reg,
        iterations = 1000,
    )
    x_approx = solve!(S, bR)
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1
    =#
end

function testVerboseSolvers(; arrayType=Array, elType=Float32)
    A = rand(elType, 3, 2)
    x = rand(elType, 2)
    b = A * x

    solvers = [ADMM, FISTA, POGM, OptISTA, SplitBregman]

    for solver in solvers
        @test try
            S = createLinearSolver(solver, arrayType(A), iterations=3, verbose=true)
            solve!(S, arrayType(b))
            true
        catch e
            @error e
            false
        end
    end
end

# CGNR started from x0 converges to the same (regularized) least-squares solution and leaves the
# data unchanged, with the forward operator and with only the normal operator
function testCGNRStartValue(; arrayType=Array, elType=Float64)
    A = randn(elType, 20, 8)   # well conditioned: CG takes at most 8 steps here
    b = randn(elType, 20)
    x0 = randn(elType, 8)
    for λ in (zero(elType), elType(0.5))
        x_ls = (A' * A + λ * I) \ (A' * b)
        S = Adapt.adapt(arrayType, createLinearSolver(CGNR, arrayType(A); reg=L2Regularization(λ), iterations=50))
        b_d = arrayType(b)
        x_approx = Array(solve!(S, b_d; x0=arrayType(x0)))
        @test x_approx ≈ x_ls rtol = sqrt(eps(elType))
        @test Array(b_d) == b

        S = Adapt.adapt(arrayType, createLinearSolver(CGNR, nothing; AHA=arrayType(A' * A), reg=L2Regularization(λ), iterations=50))
        x_approx = Array(solve!(S, arrayType(A' * b); x0=arrayType(x0)))
        @test x_approx ≈ x_ls rtol = sqrt(eps(elType))

        # the same solvers built on the CPU and moved to adaptType with Adapt.adapt
        S = Adapt.adapt(adaptType, createLinearSolver(CGNR, A; reg=L2Regularization(λ), iterations=50))
        @test Array(solve!(S, adaptType(b); x0=adaptType(x0))) ≈ x_ls rtol = sqrt(eps(elType))
        S = Adapt.adapt(adaptType, createLinearSolver(CGNR, nothing; AHA=A' * A, reg=L2Regularization(λ), iterations=50))
        @test Array(solve!(S, adaptType(A' * b); x0=adaptType(x0))) ≈ x_ls rtol = sqrt(eps(elType))
    end
end

# Everything a CGNR init! may modify, to check that a rejected init! leaves the solver unchanged
function cgnrSnapshot(S)
    states = hasproperty(S.state, :states) ? S.state.states : [S.state]
    active = hasproperty(S.state, :active) ? copy(S.state.active) : nothing
    return (objectid(S.state), objectid(S.L2), λ(S.L2), active, [(objectid(s), map(objectid, (s.x, s.x₀, s.pl, s.vl)),
        map(Array, (s.x, s.x₀, s.pl, s.vl)), s.αl, s.βl, s.ζl, s.iteration, s.z0) for s in states])
end

function testCGNRRejected(f, S, needles...)
    before = cgnrSnapshot(S)
    err = try f(); nothing catch e; e end
    @test err isa ArgumentError
    @test all(n -> occursin(n, sprint(showerror, err)), ("Adapt.adapt(target, solver)", needles...))
    @test isequal(cgnrSnapshot(S), before)
end

backendName(T) = T === Array ? "CPU" : string(nameof(T))

# A sparse vector with its entries on a JLArray device outside the GPUArrays hierarchy, like CUDA.jl 5's CuSparseVector
struct DeviceSparseVector{Tv, Ti} <: RegularizedLeastSquares.SparseArrays.AbstractSparseVector{Tv, Ti}
    nzind::JLArray{Ti, 1}
    nzval::JLArray{Tv, 1}
    len::Int
end
Base.size(v::DeviceSparseVector) = (v.len,)

# CGNR runs on the backend it was constructed on or adapted to; data or operators on another backend are rejected
# before the solver changes
function testCGNRBackend(; elType=Float32)
    A = randn(elType, 20, 8)
    b = A * randn(elType, 8)
    x_cpu = solve!(createLinearSolver(CGNR, A; iterations=8), b)
    for (target, other) in ((adaptType, Array), (Array, adaptType))
        S = Adapt.adapt(target, createLinearSolver(CGNR, A; iterations=8))
        needles = ("is on backend $(backendName(other))", "solver state is on backend $(backendName(target))")
        testCGNRRejected(() -> init!(S, other(b)), S, "b", needles...)
        testCGNRRejected(() -> solve!(S, other([b b])), S, "b", needles...)

        # views of arrays on the solver's backend use the solver's buffers
        state = S.state
        buffers = (state.x, state.x₀, state.pl, state.vl)
        init!(S, view(target([b; b]), 1:20))
        @test all(iszero, Array(S.state.x))
        @test Array(solve!(S, view(target([b; b]), 21:40))) ≈ x_cpu rtol = 2e-4
        @test S.state === state && all(map(===, (state.x, state.x₀, state.pl, state.vl), buffers))

        # an initialized state rejects b and x0 on another backend, for a vector and a matrix b
        x0 = zeros(elType, 8)
        testCGNRRejected(() -> solve!(S, other(b)), S, "b", needles...)
        testCGNRRejected(() -> solve!(S, target(b); x0=other(x0)), S, "x0", needles...)
        testCGNRRejected(() -> solve!(S, other([b b])), S, "b", needles...)
        testCGNRRejected(() -> solve!(S, target([b b]); x0=other(x0)), S, "x0", needles...)
    end

    # operators on adaptType with the default CPU state, also when b is on the backend of the state
    LO = RegularizedLeastSquares.LinearOperatorCollection.LinearOperators
    for (S, data) in ((createLinearSolver(CGNR, adaptType(A); iterations=8), b),
                      (createLinearSolver(CGNR, LO.LinearOperator(adaptType(A)); iterations=8), b),
                      (createLinearSolver(CGNR, nothing; AHA=adaptType(A' * A), iterations=8), A' * b))
        for d in (data, adaptType(data), [data data])
            testCGNRRejected(() -> solve!(S, d), S, "operator", "on backend $(backendName(adaptType))", "solver state is on backend CPU")
        end
    end

    # sparse b: host sparse arrays are on the CPU, device sparse arrays on their device
    S = createLinearSolver(CGNR, A; iterations=8)
    @test solve!(S, RegularizedLeastSquares.SparseArrays.sparse(b)) ≈ x_cpu rtol = 2e-4
    testCGNRRejected(() -> solve!(S, DeviceSparseVector(JLArray(collect(1:20)), JLArray(b), 20)), S, "b is on backend JLArray")
    # a solver on that device accepts it; only the check runs, as JLArrays has no sparse mul!
    S = Adapt.adapt(JLArray, S)
    @test isnothing(RegularizedLeastSquares.checkCGNRBackend(S, S.state, DeviceSparseVector(JLArray(collect(1:20)), JLArray(b), 20), 0))
end

# A partly iterated CGNR with a normalized, auto-scaled L2 term and a transformed projection, moved with
# Adapt.adapt, finishes with the same result as the CPU solver it was moved from
function testCGNRAdaptResume(; elType=Float64)
    A = randn(elType, 20, 8)
    b = randn(elType, 20)
    L2 = AutoScaledRegularization(L2Regularization(elType(0.1)))
    L2.factor = elType(0.5)   # a populated auto-scaling cache
    Q = Matrix(qr(randn(elType, 8, 8)).Q)
    S = createLinearSolver(CGNR, A; reg=[L2, TransformedRegularization(PositiveRegularization(), Q)],
        normalizeReg=MeasurementBasedNormalization(), iterations=6)
    init!(S, b)
    iterate(S)
    S_adapted = Adapt.adapt(adaptType, S)
    @test S_adapted.state.x isa adaptType
    @test S_adapted.constr[1].trafo isa adaptType
    @test λ(S_adapted.L2) == λ(S.L2)
    for _ in S end
    for _ in S_adapted end
    @test Array(solversolution(S_adapted)) ≈ solversolution(S) rtol = 1e-4
end

@testset "Test Solvers" begin
    @testset "$arrayType" begin
        for elType in [Float32, Float64]
            @testset "Real Linear Solver: $elType" begin
                testRealLinearSolver(; arrayType, elType)
            end

            @testset "Complex Linear Solver: $elType" begin
                testComplexLinearSolver(; arrayType, elType)
            end

            @testset "Complex Linear Solver w/ AHA Interface: $elType" begin
                testComplexLinearAHASolver(; arrayType, elType)
            end

            @testset "General Convex Solver: $elType" begin
                testConvexLinearSolver(; arrayType, elType)
            end

            @testset "CGNR with start value: $elType" begin
                testCGNRStartValue(; arrayType, elType)
            end

            @testset "CGNR backends: $elType" begin
                testCGNRBackend(; elType)
            end

            @testset "CGNR resumed after Adapt.adapt: $elType" begin
                testCGNRAdaptResume(; elType)
            end
        end
    end
    testVerboseSolvers()
end