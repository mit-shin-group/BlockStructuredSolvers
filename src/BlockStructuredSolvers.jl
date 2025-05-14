module BlockStructuredSolvers

# Import
import LinearAlgebra: I
import CUDA: @allowscalar
import AMDGPU: @allowscalar

# Export functions
export BlockTriDiagData_cpu, BlockTriDiagData_seq, BlockTriDiagData_batched
export initialize_cpu, initialize_seq, initialize_batched
export factorize!, solve!

# Include backbone files
include("backbone_cpu.jl")
include("backbone_cuda.jl")
include("backbone_rocm.jl")

# Include solver files
include("cpu.jl")
include("gpu_seq.jl")
include("gpu_batched.jl")

end
