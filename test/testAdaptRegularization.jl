# Public Adapt.adapt for regularization terms.
# CPU runs check a same-backend (Array) and a cross-backend (JLArray) target, GPU runs their own arrayType.
regTargets = arrayType === Array ? (Array, JLArray) : (arrayType,)
JLArrays.GPUArrays.allowscalar(false)

const RegLO = LinearOperatorCollection.LinearOperators
regApprox(a, b) = isapprox(a, b; rtol = 2e-4, atol = 1e-5)
regData(T, n; seed = 54) = randn(MersenneTwister(seed), T, n)
regWeights(n; seed = 55) = rand(MersenneTwister(seed), Float32, n) .+ 0.5f0
isOn(x::AbstractArray, target) = x isa target
isOn(op, target) = RegLO.storage_type(op) <: target

# the rejection is an ArgumentError naming the offending component
regRejects(f, needle) = try f(); false catch e; e isa ArgumentError && occursin(needle, sprint(showerror, e)) end

struct CustomRegTerm <: RegularizedLeastSquares.AbstractProjectionRegularization end
struct RegNoopAdaptor end
struct RegToFloat64Adaptor end
Adapt.adapt_storage(::RegToFloat64Adaptor, x::AbstractArray{Float32}) = JLArray(Float64.(x))

# prox! and norm of the adapted term on the target agree with those of the source term on the CPU
function checkProx(reg, target, x; withNorm = true)
  ref = prox!(deepcopy(reg), copy(x))
  withNorm && @test regApprox(norm(Adapt.adapt(target, reg), target(x)), norm(deepcopy(reg), x))
  moved = Adapt.adapt(target, reg)
  y = prox!(moved, target(x))
  @test y isa target
  @test regApprox(Array(y), ref)
  return moved
end

function testRegAdaptGraph(target, T)
  n = 6
  x = regData(T, n)
  D = randn(MersenneTwister(2), T, n, n) + 3I
  W = WeightingOp(T.(regWeights(n)))
  a = TransformedRegularization(L1Regularization(0.05f0), W)
  b = TransformedRegularization(AutoScaledRegularization(TransformedRegularization(a, D)), W)
  reg = NormalizedRegularization(TransformedRegularization(b, D), 1.5f0)
  D₀, w₀ = copy(D), copy(W.weights)

  moved = Adapt.adapt(target, reg)
  @test moved.factor === 1.5f0
  outer, inner = moved.reg, moved.reg.reg.reg.reg
  @test outer.trafo === inner.trafo && moved.reg.reg.trafo === inner.reg.trafo   # one rebuilt child per source child
  @test outer.trafo !== D && isOn(outer.trafo, target) && Array(outer.trafo) == D
  @test inner.reg.trafo !== W && isOn(inner.reg.trafo, target)
  @test isnothing(moved.reg.reg.reg.factor)
  checkProx(reg, target, x)
  @test D == D₀ && W.weights == w₀

  auto = AutoScaledRegularization(L1Regularization(0.05f0))
  auto.factor = 0.7f0
  @test Adapt.adapt(target, auto).factor === 0.7f0
  checkProx(auto, target, x)

  mask = [true, false, true, true, false, false]
  masked = MaskedRegularization(PositiveRegularization(), mask)
  movedMask = Adapt.adapt(target, masked).mask
  @test movedMask isa Vector{Bool} && movedMask == mask && movedMask !== mask

  w = regWeights(n)
  weighted = Adapt.adapt(target, L2Regularization(w))
  @test weighted.λ isa target{Float32} && weighted.λ !== w && Array(weighted.λ) == w

  if target === Array   # round trip through another backend
    back = Adapt.adapt(Array, Adapt.adapt(JLArray, reg))
    @test back.reg.trafo isa Matrix{T} && back.reg.trafo == D
    @test regApprox(prox!(back, copy(x)), prox!(deepcopy(reg), copy(x)))
  end
end

# Array-valued factors of weighted L2 terms
function testRegAdaptFactors(target)
  n = 6
  x = regData(Float32, n)
  w, f = regWeights(n), regWeights(n; seed = 56)
  fixed = FixedScaledRegularization(L2Regularization(w), f)
  auto = AutoScaledRegularization(L2Regularization(copy(w)))
  auto.factor = copy(f)
  for reg in (fixed, auto)
    ref = norm(reg, x)
    moved = Adapt.adapt(target, reg)
    @test moved.factor isa target{Float32} && moved.factor !== reg.factor
    @test regApprox(norm(moved, target(x)), ref)
    reg.factor .= 8
    reg.reg.λ .= 9
    @test regApprox(norm(moved, target(x)), ref)   # independent of later changes to the source
  end
end

function testRegAdaptTV(target, T)
  shape = (3, 4)
  x = regData(T, prod(shape))
  fresh = TVRegularization(0.05f0; shape, dims = 1:2, iterationsTV = 5)
  @test isnothing(Adapt.adapt(target, fresh).params)
  checkProx(fresh, target, x)
  @test isnothing(fresh.params)

  initialized = TVRegularization(0.05f0; shape, dims = [1, 2], iterationsTV = 5)
  prox!(initialized, copy(x))
  # cached gradient with a different geometry than the term's current shape/dims; prox! keeps using it
  other = TVRegularization(0.05f0; shape, dims = 1:2, iterationsTV = 5)
  other.params = RegularizedLeastSquares.TVParams(zeros(T, prod(shape)); shape = (4, 3), dims = [1])
  foreach(((i, f),) -> fill!(getfield(other.params, f), T(i)), enumerate((:pq, :rs, :pqOld, :xTmp)))
  fields = (:pq, :rs, :pqOld, :xTmp)
  for src in (initialized, other)
    before = map(f -> copy(getfield(src.params, f)), fields)
    moved = Adapt.adapt(target, src)
    for (f, b) in zip(fields, before)
      buffer = getfield(moved.params, f)
      @test buffer isa target && buffer !== getfield(src.params, f) && isequal(Array(buffer), b)
    end
    ∇, ∇′ = src.params.∇, moved.params.∇
    @test ∇′ !== ∇ && isOn(∇′, target) && size(∇′) == size(∇)
    @test regApprox(Array(∇′ * target(x)), ∇ * x)
    ∇state = gradState(∇)   # after the reference product above, which updates the source counters
    checkProx(src, target, x)
    if target === Array   # round trip through another backend against an independent CPU reference
      ref = prox!(deepcopy(src), copy(x))
      back = Adapt.adapt(Array, Adapt.adapt(JLArray, src))
      @test all(f -> getfield(back.params, f) isa Vector{T} && getfield(back.params, f) !== getfield(src.params, f), fields)
      @test regApprox(prox!(back, copy(x)), ref)
    end
    @test all(map((f, b) -> isequal(getfield(src.params, f), b), fields, before))   # source buffers untouched
    @test isequal(gradState(∇), ∇state)   # source gradient counters and workspaces untouched
  end
  @test Adapt.adapt(target, initialized).dims !== initialized.dims
  @test Adapt.adapt(target, initialized.params) isa RegularizedLeastSquares.TVParams

  # mutable host dims, including views, are copied
  d = [1, 2]
  viewDims = TVRegularization(0.1f0; shape = (2, 3), dims = view(d, 1:1))
  movedView = Adapt.adapt(target, viewDims)
  y = target(T[1, 2, 4, 8, 16, 32])
  normBefore = norm(movedView, y)
  @test movedView.dims !== viewDims.dims && movedView.dims == [1]
  d[1] = 2
  @test regApprox(norm(movedView, y), normBefore)
end

# Counters and workspaces of a gradient operator, which are mutated when it is applied
gradState(∇) = map(p -> (v = getproperty(∇, p); v isa AbstractArray ? copy(v) : v), (:nprod, :ntprod, :nctprod, :tmp, :Mv, :Mtu))

# Built-in terms run on the target without scalar indexing
function testRegAdaptExecution(target, T)
  n = 12
  x = regData(T, n)
  nested = NormalizedRegularization(AutoScaledRegularization(FixedScaledRegularization(L1Regularization(0.1f0), 2f0)), 0.5f0)
  checkProx(nested, target, x)
  checkProx(FixedParameterRegularization(L1Regularization(0.1f0)), target, x)

  mask = Bool[1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
  for inner in (RealRegularization(), PositiveRegularization(), L1Regularization(0.1f0)), m in (mask, falses(n))
    checkProx(MaskedRegularization(inner, collect(m)), target, x)
  end

  l21 = L21Regularization(0.1f0; slices = 2)
  checkProx(l21, target, x)
  @test regApprox(norm(l21, target(x)), norm(l21, x))

  llr = LLRRegularization(0.1f0; shape = (2, 3), blockSize = (2, 2), randshift = false)
  checkProx(llr, target, x; withNorm = false)   # LLR norm is CPU Vector only

  # weighted L2 norm: element i is weighted with λ[i]
  w = regWeights(n + 3)
  ref = sum(w[1:n] .* abs2.(x))
  for λ in (w[1:n], reshape(w[1:n], 3, 4), w)
    @test regApprox(norm(Adapt.adapt(target, L2Regularization(λ)), target(x)), ref)
    @test regApprox(norm(L2Regularization(λ), x), ref)
  end
  @test_throws BoundsError norm(L2Regularization(w[1:n-2]), x)
  @test_throws BoundsError norm(Adapt.adapt(target, L2Regularization(w[1:n-2])), target(x))
end

# The example of the Adapt.adapt docstring, with `target` in place of CuArray
function testRegAdaptExample(target)
  reg = AutoScaledRegularization(TransformedRegularization(L1Regularization(0.1f0), WeightingOp(rand(MersenneTwister(57), Float32, 1024))))
  gpu_reg = Adapt.adapt(target, reg)
  x = randn(MersenneTwister(58), Float32, 1024)
  y = prox!(gpu_reg, target(x))
  @test y isa target && regApprox(Array(y), prox!(deepcopy(reg), copy(x)))
end

function testRegAdaptErrors(target)
  n = 6
  D = randn(MersenneTwister(3), Float32, n, n)
  rejects = regRejects
  mask = [true, false, true, true, false, false]
  @test rejects(() -> Adapt.adapt(target, MaskedRegularization(CustomRegTerm(), mask)), "CustomRegTerm")
  opaque = AutoScaledRegularization(TransformedRegularization(L1Regularization(0.1f0), RegLO.LinearOperator(D)))
  trafo = opaque.reg.trafo
  @test rejects(() -> Adapt.adapt(target, opaque), "no Adapt rule")
  @test isnothing(opaque.factor) && opaque.reg.trafo === trafo
  @test rejects(() -> Adapt.adapt(target, TransformedRegularization(L1Regularization(0.1f0), identity)), "transform of type")
  for bad in (Int, AbstractArray, RegNoopAdaptor())
    @test rejects(() -> Adapt.adapt(bad, L2Regularization(regWeights(n))), "unsupported Adapt target")
  end
  if target === JLArray
    w = regWeights(n)
    @test rejects(() -> Adapt.adapt(RegToFloat64Adaptor(), L2Regularization(w)), "element type")
    @test w == regWeights(n)
  end
end

# Callables of projection and plug-and-play terms
const RegCST = RegularizedLeastSquares.ClampedScalingTransform
struct RegScaleProj{V} <: Function   # projection callable owning data, with an owner rule
  w::V
end
(p::RegScaleProj)(x) = x .* p.w
Adapt.adapt_structure(to, p::RegScaleProj) = RegScaleProj(Adapt.adapt(to, p.w))
struct RegModel{V}   # PnP model owning data, with an owner rule
  w::V
end
(m::RegModel)(x) = x .* reshape(m.w, size(x))
Adapt.adapt_structure(to, m::RegModel) = RegModel(Adapt.adapt(to, m.w))
struct RegFixedFactory{Tr}   # input-transform factory retaining a transform, with an owner rule
  t::Tr
end
(f::RegFixedFactory)(x) = f.t
Adapt.adapt_structure(to, f::RegFixedFactory) = RegFixedFactory(Adapt.adapt(to, f.t))
struct RegNoRuleModel{V}; w::V; end
(m::RegNoRuleModel)(x) = x
struct RegNoRuleProj{V} <: Function; w::V; end
(p::RegNoRuleProj)(x) = x
struct RegThrowingModel end
(m::RegThrowingModel)(x) = x
Adapt.adapt_structure(to, m::RegThrowingModel) = error("owner rule failed")
struct RegStatelessModel end
(m::RegStatelessModel)(x) = x
Adapt.adapt_structure(to, m::RegStatelessModel) = m

# An adaptor with rules of its own, for children with both a general and an adaptor-specific rule
struct RegCountingAdaptor
  count::Base.RefValue{Int}
end
Adapt.adapt_storage(::RegCountingAdaptor, x::Array) = JLArray(x)
struct RegCountChild{V}; w::V; end
Adapt.adapt_structure(to, c::RegCountChild) = RegCountChild(Adapt.adapt(to, c.w))
Adapt.adapt_structure(a::RegCountingAdaptor, c::RegCountChild) = (a.count[] += 1; RegCountChild(JLArray(2 .* c.w)))
struct RegCountModel{V}; w::V; end
(m::RegCountModel)(x) = x .* reshape(m.w, size(x))
Adapt.adapt_structure(to, m::RegCountModel) = RegCountModel(Adapt.adapt(to, m.w))
Adapt.adapt_structure(a::RegCountingAdaptor, m::RegCountModel) = (a.count[] += 1; RegCountModel(JLArray(2 .* m.w)))
struct RegHolder{C}; child::C; end
(h::RegHolder)(x) = x .* reshape(h.child.w, size(x))
Adapt.adapt_structure(to, h::RegHolder) = RegHolder(Adapt.adapt(to, h.child))
struct RegCountArr{V} <: AbstractVector{Float32}; data::V; end
Base.size(a::RegCountArr) = size(a.data)
Base.getindex(a::RegCountArr, i::Int) = a.data[i]
Adapt.adapt_structure(to, a::RegCountArr) = RegCountArr(Adapt.adapt(to, a.data))
Adapt.adapt_structure(a::RegCountingAdaptor, c::RegCountArr) = (a.count[] += 1; JLArray(2 .* c.data))

regScale(v, s) = y -> y .* v .* s          # closure capturing a plain array and a number
regHold(c) = y -> y .* reshape(c.w, size(y))   # closure capturing a child with its own rules
regCounting(c) = y -> (c[] += 1; y)        # closure capturing external mutable state
function regBoxed(n)                       # closure capturing a reassigned variable
  w = regWeights(n)
  f = y -> y .* w
  w = regWeights(n; seed = 70)
  return f
end

function testRegAdaptCallables(target, T)
  n = 12
  x = regData(T, n)
  w = regWeights(n)
  for proj in (ProjectionRegularization(), ProjectionRegularization(projFunc = regScale(copy(w), 2f0)), ProjectionRegularization(projFunc = RegScaleProj(copy(w))))
    moved = checkProx(proj, target, x)
    if target === Array
      back = Adapt.adapt(Array, Adapt.adapt(JLArray, proj))
      @test regApprox(prox!(back, copy(x)), prox!(deepcopy(proj), copy(x)))
    end
    f, f′ = proj.projFunc, moved.projFunc
    if f isa RegScaleProj && target !== Array
      @test f′.w isa target
      f′.w .= 0   # mutating owned data of the adapted term leaves the source alone
      @test f.w == w
    elseif !(f isa RegScaleProj) && fieldcount(typeof(f)) > 0
      @test f′.v isa target && f′.v !== f.v && f′.s === 2f0   # plain captures get their own copies, also on Array
      f′.v .= 0
      @test f.v == w
    end
  end
  checkProx(MaskedRegularization(ProjectionRegularization(), Bool[1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0]), target, x)

  c = Ref(0)   # captured mutable state keeps Adapt's semantics: shared, not cloned
  prox!(Adapt.adapt(target, ProjectionRegularization(projFunc = regCounting(c))), target(x))
  @test c[] == 1

  shape = [3, 4]
  model = RegModel(regWeights(n; seed = 60))
  data = reshape(Float32[-1, 0.5, 0, 2, 0.25, 1, -0.5, 0.75, 3, 0.1, 1, 0.9], 3, 4)
  clamped = RegCST(data, 0f0, 1f0)
  snapshot = (copy(model.w), copy(clamped.mask), copy(clamped.x))
  for factory in (RegularizedLeastSquares.IdentityTransform, RegularizedLeastSquares.MinMaxTransform, RegularizedLeastSquares.ZTransform, RegFixedFactory(clamped)), ignoreIm in (false, true)
    pnp = PlugAndPlayRegularization(0.5f0; model, shape, input_transform = factory, ignoreIm)
    moved = checkProx(pnp, target, x; withNorm = false)   # PnP has no norm
    @test moved.λ === 0.5f0 && moved.ignoreIm == ignoreIm && moved.shape == shape && moved.shape !== pnp.shape
    if factory isa RegFixedFactory
      t′ = moved.input_transform.t
      @test t′ isa RegCST && t′.mask isa target && t′.x isa target && t′.x !== clamped.x
    else
      @test moved.input_transform === factory
    end
    if target === Array
      back = Adapt.adapt(Array, Adapt.adapt(JLArray, pnp))
      @test regApprox(prox!(back, copy(x)), prox!(deepcopy(pnp), copy(x)))
    else
      @test moved.model.w isa target
      moved.model.w .= 0
    end
    @test all(map(isequal, snapshot, (model.w, clamped.mask, clamped.x)))   # source untouched
  end
end

# Transform objects, and the restoration of clamped values by the elementwise inverse
function testRegAdaptTransforms(target)
  inv, fwd = RegularizedLeastSquares.inverse_transform, RegularizedLeastSquares.transform
  # below minimum, minimum, interior, maximum and above maximum
  data = Float32[-1, 0, 0.3, 1, 2, 0.7]
  t = RegCST(data, 0f0, 1f0)
  t′ = Adapt.adapt(target, t)
  @test t′.mask isa target && t′.x isa target && t′.x !== t.x && Array(t′.mask) == t.mask
  @test regApprox(Array(fwd(t′, target(data))), fwd(t, data))
  z = rand(MersenneTwister(62), Float32, 6)
  old(out, t) = (out = copy(out); out[t.mask] = t.x[t.mask]; out)   # the previous masked restoration
  ref = old(z, t)
  @test inv(t, z) == ref && ref[t.mask] == data[t.mask] && t.mask == Bool[1, 0, 0, 1, 1, 0]
  @test regApprox(Array(inv(t′, target(z))), ref)
  # a mask vector over a matrix restores by linear index, an incompatible mask shape is out of bounds
  z23 = reshape(z, 2, 3)
  @test inv(t, z23) == old(z23, t)
  @test regApprox(Array(inv(t′, target(z23))), old(z23, t))
  t31 = RegCST(reshape(Float32[-1, 0.5, 2], 3, 1), 0f0, 1f0)
  z32 = rand(MersenneTwister(63), Float32, 3, 2)
  @test_throws BoundsError old(z32, t31)
  @test_throws BoundsError inv(t31, z32)
  @test_throws BoundsError inv(Adapt.adapt(target, t31), target(z32))

  zt = RegularizedLeastSquares.ZTransform(regWeights(6), regWeights(6; seed = 64))
  zt′ = Adapt.adapt(target, zt)
  @test zt′.mean isa target && zt′.mean !== zt.mean && regApprox(Array(fwd(zt′, target(data))), fwd(zt, data))
  mm = RegularizedLeastSquares.MinMaxTransform(data)
  @test Adapt.adapt(target, mm) === mm   # scalar fields
end

function testRegAdaptCallableErrors(target)
  n = 4
  x = regData(Float32, n)
  pnp(model) = PlugAndPlayRegularization(0.5f0; model, shape = [n])
  @test regRejects(() -> Adapt.adapt(target, pnp(RegNoRuleModel(regWeights(n)))), "RegNoRuleModel")
  @test regRejects(() -> Adapt.adapt(target, ProjectionRegularization(projFunc = RegNoRuleProj(regWeights(n)))), "RegNoRuleProj")
  @test regRejects(() -> Adapt.adapt(target, ProjectionRegularization(projFunc = Base.Experimental.@opaque y -> y)), "opaque closure")
  boxed = regBoxed(n)
  @test fieldtypes(typeof(boxed)) == (Core.Box,)
  @test regRejects(() -> Adapt.adapt(target, ProjectionRegularization(projFunc = boxed)), "Core.Box")
  @test regRejects(() -> Adapt.adapt(target, MaskedRegularization(ProjectionRegularization(projFunc = boxed), trues(n))), "ProjectionRegularization.projFunc")
  @test_throws ErrorException Adapt.adapt(target, pnp(RegThrowingModel()))   # errors of an owner rule propagate
  stateless = RegStatelessModel()
  @test Adapt.adapt(target, pnp(stateless)).model === stateless
  s = CGNR(randn(Float32, n, n); reg = L2Regularization(0.1f0))   # CGNR's own gate still rejects projections
  @test regRejects(() -> Adapt.adapt(target, CGNR(s.A, s.AHA, s.L2, [ProjectionRegularization()], s.normalizeReg, s.iterations, s.state)), "ProjectionRegularization")

  if target === JLArray   # rules specific to the user's adaptor apply once, at every level
    w = regWeights(n)
    for model in (RegCountModel(copy(w)), RegHolder(RegCountChild(copy(w))))
      a = RegCountingAdaptor(Ref(0))
      m′ = Adapt.adapt(a, pnp(model)).model
      @test a.count[] == 1 && Array(m′ isa RegHolder ? m′.child.w : m′.w) == 2 .* w
    end
    a = RegCountingAdaptor(Ref(0))
    moved = Adapt.adapt(a, ProjectionRegularization(projFunc = regHold(RegCountChild(copy(w)))))
    @test a.count[] == 1 && Array(moved.projFunc.c.w) == 2 .* w
    a = RegCountingAdaptor(Ref(0))
    v = view(RegCountArr(Float32[1, 2]), :)
    moved = Adapt.adapt(a, ProjectionRegularization(projFunc = regScale(v, 1f0)))
    @test a.count[] == 1 && Array(moved.projFunc.v) == Float32[2, 4] && v == Float32[1, 2]
  end
end

@testset "Regularization Adapt" begin
  for target in regTargets
    @testset "$target" begin
      for T in (Float32, ComplexF32)
        @testset "graph $T" begin testRegAdaptGraph(target, T) end
        @testset "TV $T" begin testRegAdaptTV(target, T) end
        @testset "execution $T" begin testRegAdaptExecution(target, T) end
        @testset "callables $T" begin testRegAdaptCallables(target, T) end
      end
      @testset "array factors" begin testRegAdaptFactors(target) end
      @testset "docstring example" begin testRegAdaptExample(target) end
      @testset "errors" begin testRegAdaptErrors(target) end
      @testset "transforms" begin testRegAdaptTransforms(target) end
      @testset "callable errors" begin testRegAdaptCallableErrors(target) end
    end
  end
end
