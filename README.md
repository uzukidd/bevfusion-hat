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

3. Compile `qat.bc` into `.HBM`:

   ```bash
   bash scripts/compile_s600.sh --opt 1 --jobs 4
   ```

4. (Optional) If you want to compile `qat.bc` into `.HBIR` without generating `.HBM`:

   ```bash
   bash scripts/compile_s600.sh --convert-only
   ```

5. Run HBM demo:

   ```bash
   bash scripts/run_demo.sh
   ```

6. Run the verification:

   ```bash
   bash scripts/oe_tools.sh verify_float
   bash scripts/oe_tools.sh verify_hbir
   ```
