# BEVFusion HAT for RDK S600

## Requirements
- Linux or WSL
- python 3.10
- pytorch 2.8.0 with CUDA 12.8
- horizon-plugin-pytorch 3.1.5 with CUDA 12.8
- hdbk 4.7.5

See [requirements.txt](.\requirements.txt) for more details.

## Usages

1. Download float checkpoint:

   ```bash
   bash ./scripts/oe_tools.sh oe_assets --model-assets --float-checkpoint
   ```

   ```
   float-checkpoint-last.pth.tar
       └─→ Pytorch float model (used only for verification)
   
   qat.bc
       └─→ Perform fixed-point conversion targeting S600's nash-p
               └─→ Generate bevfusion.bc locally
                       ├─→ Run HBIR inference locally
                       └─→ Compile to .hbm
   
   quantized.bc (official nash-m version)
       └─→ Run fixed-point inference locally as a comparison baseline
   ```

2. Run float model demo:

   ```bash
   bash scripts/run_float.sh --samples data/samples
   ```

3. Compile `qat.bc` to `.HBM`:

   ```bash
   bash scripts/compile_s600.sh --opt 1 --jobs 4
   ```

4. (Optional) If you want to compile `qat.bc` to `.HBIR` without generating `.HBM`:

   ```bash
   bash scripts/compile_s600.sh --convert-only
   ```

5. Run S600 nash-p HBIR demo:

   ```bash
   bash scripts/run_demo.sh
   ```

## Benchmarking

Run the benchmark test as follows:

```bash
# pyTorch float model
bash scripts/benchmark_models.sh --models float --warmup 5 --iterations 100
bash scripts/oe_tools.sh verify_float

# nash-m HBIR
bash scripts/benchmark_models.sh --models official --warmup 3 --iterations 20

# S600 nash-p HBIR
bash scripts/benchmark_models.sh --models s600 --warmup 3 --iterations 20
bash scripts/oe_tools.sh verify_hbir
```

| Model   | mean        | mediam      | p95         | serial FPS | `occupancy_argmax_agreement` |
| ------- | ----------- | ----------- | ----------- | ---------- | ---------------------------- |
| floa    | 53.677 ms   | 51.396 ms   | 66.112 ms   | 18.63      | 0.93                         |
| nash-m* | 986.060 ms  | 963.104 ms  | 162.958 ms  | 1.014      | -                            |
| nash-p* | 6509.658 ms | 6470.506 ms | 6807.733 ms | 0.154      | 0.98                         |

*The execution time of the HBIR interpreter on the CPU cannot be used to conclude that the S600 hardware is slower; actual hardware performance must be measured by running HBM.
