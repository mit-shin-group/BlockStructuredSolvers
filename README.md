# TBD-GPU benchmark runs

## Install dependencies

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=./benchmark -e 'using Pkg; Pkg.add(path="."); Pkg.instantiate()'
```

### Run benchmarks

```bash
julia --project=./benchmark benchmark/benchmark_cpu.jl
julia --project=./benchmark benchmark/benchmark_gpu.jl
julia --project=./benchmark benchmark/benchmark_kf.jl
julia --project=./benchmark benchmark/benchmark_kf_plot.jl
```
