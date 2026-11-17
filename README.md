# BEVFusion HAT for RDK S600

## Requirements
- Linux or WSL
- python 3.10
- pytorch 2.8.0 with CUDA 12.8
- horizon-plugin-pytorch 3.1.5 with CUDA 12.8
- hdbk 4.7.5

See [requirements.txt](.\requirements.txt) for more details.

## Usages

1. Download the float checkpoint and HBIR assets directly:

   ```bash
   bash scripts/download_models.sh
   ```

   

   Downloaded assets are stored as follows.

   ```text
   data/checkpoints/
   ├── float-checkpoint-last.pth.tar  # PyTorch float weights: inference or local calibration/export
   ├── qat.bc                        # Official HBIR with quantization info, before target lowering
   ├── quantized.bc                  # Official nash-m fixed-point HBIR: inference/comparison
   └── SHA256SUMS                    # Locally computed SHA-256 hashes
   ```

   - **[Float checkpoint](data/checkpoints/float-checkpoint-last.pth.tar):** load with the PyTorch model for float inference, or calibrate and export a new HBIR before nash-p conversion and HBM compilation.
   - **[qat.bc](data/checkpoints/qat.bc):** already contains quantization information; it is not an unquantized float model. Use it as the source for nash-p fixed-point conversion, then compile the result to S600 HBM. The filename does not imply QAT training.
   - **[quantized.bc](data/checkpoints/quantized.bc):** already lowered for nash-m. Use it for host inference or as a reference; S600 retargeting requires the corresponding pre-lowering source HBIR, rather than directly converting this file.

2. Run pytorch model demo:

   ```bash
   bash scripts/run_float.sh --samples data/samples
   ```

### Convert float models to HBIR


1. Export calibration samples

   ```bash
   bash scripts/export_calibration.sh \
     --samples data/samples \
     --output outputs/calibration
   ```


2. Compile pytorch model to nash-p `.HBM`:

   ```bash
   bash scripts/compile_pytorch.sh \
     --checkpoint data/checkpoints/float-checkpoint-last.pth.tar \
     --calibration-manifest outputs/calibration/manifest.json \
     --format hbm \
     --opt 1 \
     --jobs 4 \
     --output outputs/hbm_from_pytorch
   ```

3. (Optional) If you want to compile pytorch to `.HBIR` without generating `.HBM`:

   ```bash
   bash scripts/compile_pytorch.sh \
     --checkpoint data/checkpoints/float-checkpoint-last.pth.tar \
     --calibration-manifest outputs/calibration/manifest.json \
     --format hbir \
     --output outputs/hbir_from_pytorch
   ```

4. Run nash-p HBIR demo:

   ```bash
   bash scripts/run_demo.sh \
     --model outputs/hbir_from_pytorch/bevfusion.bc \
     --output outputs/hbir_from_pytorch_validation
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
