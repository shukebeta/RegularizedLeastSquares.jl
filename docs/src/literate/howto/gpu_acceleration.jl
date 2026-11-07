# # GPU Acceleration
gpu = Array; #hide
# RegularizedLeastSquares.jl supports generic GPU acceleration. This means that the user can use any GPU array type that supports the GPUArrays interface. This includes [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), and [Metal.jl](https://github.com/JuliaGPU/Metal.jl).
# To perform a reconstruction on the GPU, one has to load a GPU backend package such as CUDA and specify the GPU array type:
# ```julia
# using CUDA
# gpu = CuArray
# ```

# At first we will look at an example of dense GPU arrays.
using RegularizedLeastSquares
A = gpu(rand(Float32, 32, 16))
x = gpu(rand(Float32, 16))
b = A*x;

# ## Moving a CGNR Solver with Adapt.jl
# A `CGNR` solver is moved to the GPU as a whole with [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl). We construct it on the CPU, and `Adapt.adapt` returns a new solver
# whose operators, regularization and state all live on the target backend, and leaves the original solver untouched:
using Adapt
cpu_solver = createLinearSolver(CGNR, Array(A); reg = L2Regularization(0.0001), iterations=32);
gpu_solver = Adapt.adapt(gpu, cpu_solver);

# The moved solver is initialized and iterated with data on the GPU, which is what `solve!(gpu_solver, b)` does as well:
init!(gpu_solver, b)
while !isnothing(iterate(gpu_solver)) end
x_approx = solversolution(gpu_solver)

# To compare with the CPU result, we solve with the original CPU solver and CPU data:
x_cpu = solve!(cpu_solver, Array(b))
isapprox(Array(x_approx), x_cpu)

# `CGNR` does not change backends on its own. If its operators, its state, `b` or an array-valued `x0` live on different backends, `init!` and `solve!` throw an `ArgumentError`
# before modifying the solver, for example when a CPU solver receives a GPU `b`, or when a `CGNR` constructed from a GPU matrix still has its default CPU state.
# Views of arrays on the solver's backend are accepted.

# A matrix of measurements is solved with one state per column on the solver's backend, and a later vector reuses the state of the first column:
X = gpu(rand(Float32, 16, 3))
B = A*X
X_approx = solve!(gpu_solver, B)
x_first = solve!(gpu_solver, B[:, 1])
isapprox(Array(X_approx[:, 1]), Array(x_first))

# This works for dense matrices and for LinearOperatorCollection.jl operators that support Adapt.jl, including the default normal operator CGNR builds for them.
# Initialized solvers keep their progress, including the multi-frame states created by matrix-valued `solve!`. Unsupported operators, such as closure-based
# `LinearOperator`s, raise an `ArgumentError` instead of producing a solver that mixes backends.

# `FISTA`, `OptISTA` and `POGM` are moved in the same way, also after they have started iterating: the adapted solver
# resumes where the source stopped, with the same step size, momentum and iteration count, and the source is not modified.
cpu_fista = createLinearSolver(FISTA, Array(A); reg = L1Regularization(0.0001f0), iterations = 32);
init!(cpu_fista, Array(b))
for _ in 1:5
  iterate(cpu_fista)
end
gpu_fista = Adapt.adapt(gpu, cpu_fista)
for _ in 1:5
  iterate(gpu_fista)
end
solversolution(gpu_fista);
# Like `CGNR`, these solvers do not change backends on their own: if their operators, their state, `b` or an array-valued `x0` live on different backends,
# `init!` and `solve!` throw an `ArgumentError` before modifying the solver. This includes a solver constructed from GPU operators, whose state is
# allocated on the CPU, so move it with `Adapt.adapt` first. Views of arrays on the solver's backend are accepted, and a matrix `b` creates one state
# per column on the solver's backend that a later vector `b` reuses, as for `CGNR`.
# For a LinearOperatorCollection.jl operator `A`, these solvers default to the opaque product `AHA = A' * A`, which cannot be moved;
# construct them with `AHA = normalOperator(A)` instead.

# Other solvers (`Kaczmarz`, `ADMM`, `SplitBregman` and the direct solvers) do not support `Adapt.adapt` yet and allocate their states based
# on the type of the given measurement vector; for them, construct the solver from operators that already live on the GPU, as shown below.

# Regularization terms can be moved on their own, for example to pass them to a solver built from GPU operators.
# `Adapt.adapt(gpu, reg)` rebuilds the built-in terms and wrappers with their effective state, such as an initialized
# `AutoScaledRegularization` factor or the buffers of a `TVRegularization`, and shares no data with `reg`:
reg = AutoScaledRegularization(L2Regularization(0.0001))
gpu_reg = Adapt.adapt(gpu, reg)
# Projection and plug-and-play terms are moved with the Adapt rules of their callables. Closures are rebuilt by Adapt.jl; a callable
# type that holds data, such as a learned model, needs its own `Adapt.adapt_structure` rule, which decides what it moves:
struct Shrink{V}
  w::V
end
(s::Shrink)(x) = x .* reshape(s.w, size(x))
Adapt.adapt_structure(to, s::Shrink) = Shrink(Adapt.adapt(to, s.w))
pnp = PlugAndPlayRegularization(0.5f0; model = Shrink(rand(Float32, 16)), shape = [4, 4])
gpu_pnp = Adapt.adapt(gpu, pnp)
prox!(gpu_pnp, gpu(rand(Float32, 16)));
# Global variables used by these functions are not moved. Opaque closures, closures capturing a reassigned variable and callables
# without an Adapt rule raise an `ArgumentError`, as do unsupported terms and transforms inside a supported term, e.g. a closure-based
# `LinearOperator`.

# ## Matrix-Free Operators
# A special case is the usage of matrix-free operators. Since these operators do not have a concrete matrix representation, their GPU support depends on their implementation.
# Since not all multiplications within a solver approximation are in-place, the operator also needs to support the `*` operation and construct an appropriate result vector.
# For matrix-free operators based on LinearOperators.jl, this can be achieved by implementing the `LinearOperators.storage_type` method.

# In the following, we will take another look at the CS example and execute it on the GPU. Note that for the JLArray example we chose a small phantom, since the JLArray implementation is not optimized for performance:
using ImagePhantoms, ImageGeoms
N = 32
image = shepp_logan(N, SheppLoganToft())

using Random, LinearOperatorCollection
randomIndices = shuffle(eachindex(image))
sampledIndices = sort(randomIndices[1:div(end, 3)]);

# To construct the operator, we need to convert the indices to a GPU array.
# We also need to specify the correct storage type. In both LinearOperators.jl and LinearOperatorCollection.jl this is done with the `S` keyword argument.
gpu_indices = gpu(sampledIndices)
A = SamplingOp(eltype(image), pattern = gpu_indices, shape = size(image), S = typeof(b));

# Let's inspect the storage type of the operator:
using LinearOperatorCollection.LinearOperators
LinearOperators.storage_type(A)

# Afterwards we can use the operator as usual:
b = A*vec(gpu(image));

# And use it in the solver. `FISTA` allocates its state on the CPU and does not change backends on its own, so we construct it with the normal operator
# `normalOperator(A)`, which Adapt.jl can move, and move the solver to the GPU before solving:
using RegularizedLeastSquares
reg = TVRegularization(0.01; shape=size(image))
solver = createLinearSolver(FISTA, A; AHA=normalOperator(A), reg=reg, iterations=20)
solver = Adapt.adapt(gpu, solver)
img_approx = solve!(solver,b);

# To visualize the reconstructed image, we need to reshape the result vector to the correct shape and convert it to a CPU array:
img_approx = reshape(Array(img_approx),size(image))

# We will again use CairoMakie for visualization:
using CairoMakie
function plot_image(figPos, img; title = "", width = 150, height = 150)
  ax = CairoMakie.Axis(figPos; yreversed=true, title, width, height)
  hidedecorations!(ax)
  heatmap!(ax, img)
end
fig = Figure()
plot_image(fig[1,1], image, title = "Image")
samplingMask = fill(false, size(image))
samplingMask[sampledIndices] .= true
plot_image(fig[1,2], image .* samplingMask, title = "Sampled Image")
plot_image(fig[1,3], img_approx, title = "Reconstructed Image")
resize_to_layout!(fig)
fig