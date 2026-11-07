function testMultiThreadingSolver(; arrayType=Array, scheduler=MultiDataState)
  A = rand(ComplexF32, 3, 2)
  x = rand(ComplexF32, 2, 4)
  b = A * x

  solvers = linearSolverList()
  @testset "$(solvers[i])" for i = 1:length(solvers)
    S = createLinearSolver(solvers[i], arrayType(A), iterations=100)
    # CGNR does not switch backends implicitly
    solvers[i] <: CGNR && (S = Adapt.adapt(arrayType, S))

    x_sequential = hcat([Array(solve!(S, arrayType(b[:, j]))) for j = 1:size(b, 2)]...)
    @test x_sequential ≈ x rtol = 0.1

    x_approx = Array(solve!(S, arrayType(b), scheduler=scheduler))
    @test x_approx ≈ x rtol = 0.1

    # Does sequential/normal reco still works after multi-threading
    x_vec = Array(solve!(S, arrayType(b[:, 1])))
    @test x_vec ≈ x[:, 1] rtol = 0.1
  end

  @testset "CGNR via Adapt" begin
    # built on the CPU and moved to adaptType with Adapt.adapt
    S = Adapt.adapt(adaptType, createLinearSolver(CGNR, A, iterations=100))
    @test Array(solve!(S, adaptType(b), scheduler=scheduler)) ≈ x rtol = 0.1
    @test Array(solve!(S, adaptType(b[:, 1]))) ≈ x[:, 1] rtol = 0.1
  end

  @testset "CGNR frames on the adapted backend" begin
    # vector -> matrix -> vector with a start value matches the CPU solver per column
    A_wc = randn(ComplexF32, 20, 8)
    X = randn(ComplexF32, 8, 3)
    B = A_wc * X
    x0 = randn(ComplexF32, 8)
    S_cpu = createLinearSolver(CGNR, A_wc, iterations=8)
    X_cpu = hcat([copy(solve!(S_cpu, B[:, j]; x0)) for j = 1:3]...)
    S = Adapt.adapt(adaptType, createLinearSolver(CGNR, A_wc, iterations=8))
    @test Array(solve!(S, adaptType(B[:, 2]); x0=adaptType(x0))) ≈ X_cpu[:, 2] rtol = 2e-4
    B_d = adaptType(B)
    init!(S, B_d; x0=adaptType(x0), scheduler=scheduler)
    @test S.state isa scheduler
    @test all(s -> all(v -> v isa adaptType, (s.x, s.x₀, s.pl, s.vl)) && Array(s.x) == x0, S.state.states)
    for _ in S end
    @test Array(solversolution(S)) ≈ X_cpu rtol = 2e-4
    @test Array(B_d) == B
    first_state = first(S.state.states)
    @test Array(solve!(S, adaptType(B[:, 3]); x0=adaptType(x0))) ≈ X_cpu[:, 3] rtol = 2e-4
    @test S.state === first_state

    # b or x0 on another backend is rejected before a frame transition, in both directions
    init!(S, B_d; scheduler=scheduler)
    iterate(S)
    testCGNRRejected(() -> init!(S, B[:, 1]), S, "b is on backend CPU")
    testCGNRRejected(() -> init!(S, adaptType(B[:, 1]); x0), S, "x0 is on backend CPU")
    testCGNRRejected(() -> init!(S, B; scheduler=scheduler), S, "b is on backend CPU")
    S = Adapt.adapt(adaptType, createLinearSolver(CGNR, A_wc, iterations=8))
    testCGNRRejected(() -> init!(S, B; scheduler=scheduler), S, "b is on backend CPU")
    testCGNRRejected(() -> init!(S, B_d; x0, scheduler=scheduler), S, "x0 is on backend CPU")
  end

  @testset "CGNR resumed after Adapt.adapt" begin
    # a partly iterated matrix solve keeps its scheduler state and progress when moved
    S = createLinearSolver(CGNR, A, iterations=100)
    init!(S, b; scheduler=scheduler)
    iterate(S)
    S.state.active[2] = false   # a frame the scheduler no longer iterates
    S_adapted = Adapt.adapt(adaptType, S)
    @test S_adapted.state isa scheduler
    @test S_adapted.state.active == [true, false, true, true]
    @test S_adapted.state.states[1].iteration == 1
    for _ in S end
    for _ in S_adapted end
    @test Array(solversolution(S_adapted)) ≈ solversolution(S) rtol = 1e-4
  end
end

@testset "Test MultiThreading Support" begin
  @testset "$arrayType" begin
    for scheduler in [SequentialState, MultiThreadingState]
      @testset "$scheduler" begin
        testMultiThreadingSolver(; arrayType, scheduler)
      end
    end
  end
end