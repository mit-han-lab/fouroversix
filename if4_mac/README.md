# IF4 / NVFP4 MAC RTL Release

This directory contains the RTL for the IF4 MAC and NVFP4 MAC blocks, along with a simple testbench and golden-reference generator used to check the outputs.

## Contents

- `rtl/if4_mac/`: IF4 MAC RTL source code
- `rtl/nvfp4_mac/`: NVFP4 MAC RTL source code
- `tb/`: IF4 and NVFP4 testbenches
- `scripts/gen_golden_ref.py`: regenerates the simple golden-reference vectors
- `scripts/run_testbench.sh`: example Xcelium run script

## External Dependency

The RTL instantiates `CW_fp_add` and `CW_fp_mult`, which are floating-point adder and multiplier IP blocks. Their simulation models are not included here.

To run the testbenches, set:

```bash
export CW_SIM_DIR=/path/to/chipware/sim/verilog/CW
```

That directory must contain:

- `CW_fp_add.v`
- `CW_fp_mult.v`

## Regenerate Golden Reference

```bash
python scripts/gen_golden_ref.py
```

## Run Testbenches

```bash
./scripts/run_testbench.sh if4
./scripts/run_testbench.sh nvfp4
```

## Notes

- The golden-reference vectors are deterministic, so rerunning the generator should reproduce the same files.
- The final accumulated output is produced when the next segment begins, so the testbench includes one final flush transaction at the end.
