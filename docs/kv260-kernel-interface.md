# KV260 kernel interface

## Scope

`ddit_block_kv260_axi_top.sv` is the vendor boundary for one autonomous MDLM
block. It exposes one AXI4 memory master and one AXI4-Lite control slave. The
kernel loads a Q10 residual canvas from DDR, preloads constants, executes the
real block image with internal norm1, and writes the final Q10 canvas to DDR.

This boundary elaborates and its AXI-Lite register behavior is simulated on
macOS. Vivado synthesis, placement, timing, and KV260 execution remain open
because AMD vendor tools are not installed on this host.

The complete sequencer also passes a reduced compute-shape DMA test. It loads
all 2,048 residual records, runs autonomous norm1 plus the reduced DDiT block,
and drains output through the AXI write response in 135,051 cycles. Residual
DMA and constant preload are active concurrently through the shared read
arbiter. That first optimization reduced 152,127 cycles to 148,957. Grouped
attention projection and four-lane requantization then remove another 13,906
cycles. The combined reduction is 17,076 cycles, or 11.2248 percent. The
memory model still accounts for 6,324 reads,
1,191,808 read bytes, and 72 output bytes. A deferred Yosys hierarchy check
resolves the production AXI top successfully.

## Memory buffers

All three base addresses must be at least 128-byte aligned.

| Buffer | Required bytes | Record order | Record format |
| --- | ---: | --- | --- |
| Block image | 9,064,448 | fixed execution-image sections | `block_00_execution.bin` |
| Residual input | 262,144 | group, then six-channel tile | 2,048 records by 128 bytes |
| Final output | 262,144 allocated, 147,456 valid | six-channel tile, then group | 2,048 records by 128 bytes |

Each residual or output record stores one 576-bit vector. Bytes 0 through 71
contain 24 signed Q13.10 values in little-endian packed order. The order inside
a record is four token lanes, then six output lanes. Bytes 72 through 127 are
padding. Input record index is `group * 128 + tile`. Output record index is
`tile * 16 + group`.

The residual reader transfers both 64-byte beats, so it reads 262,144 bytes.
The output writer uses all byte strobes on the first beat and only the low eight
strobes on the second beat, so it writes 147,456 payload bytes.

## AXI-Lite registers

| Offset | Access | Meaning |
| ---: | --- | --- |
| `0x00` | R/W | status and control |
| `0x10` | R/W | block image base, low 32 bits |
| `0x14` | R/W | block image base, high 32 bits |
| `0x18` | R/W | residual input base, low 32 bits |
| `0x1c` | R/W | residual input base, high 32 bits |
| `0x20` | R/W | output base, low 32 bits |
| `0x24` | R/W | output base, high 32 bits |
| `0x28` | R | kernel cycles, low 32 bits |
| `0x2c` | R | kernel cycles, high 32 bits |
| `0x30` | R | total AXI read transactions, low 32 bits |
| `0x34` | R | residual read bytes, low 32 bits |
| `0x38` | R | constant-image bytes, low 32 bits |
| `0x3c` | R | dense-image bytes, low 32 bits |
| `0x40` | R | output bytes written, low 32 bits |
| `0x44` | R | output write transactions, low 32 bits |

Status register bits are:

| Bit | Read meaning | Write-one action |
| ---: | --- | --- |
| 0 | ready | start |
| 1 | busy | clear done |
| 2 | done sticky | clear error |
| 3 | error sticky | none |

Write all three addresses before starting. Poll done and error. Clear done and
error before reusing the kernel.

## macOS checks

```bash
.venv/bin/python -m pytest -q \
  tests/test_axi512_residual_canvas_reader_rtl.py \
  tests/test_axi512_output_canvas_writer_rtl.py \
  tests/test_ddit_block_kv260_kernel_core_rtl.py \
  tests/test_ddit_block_kv260_axi_top_rtl.py

DIFFUSION_ACCEL_RUN_KV260_CORE_RTL=1 \
  .venv/bin/python -m pytest -q \
  tests/test_ddit_block_kv260_kernel_core_functional_rtl.py
```

## Vendor synthesis gate

From a shell with Vivado available:

```bash
vivado -mode batch -source fpga/kv260/run_synth.tcl \
  -tclargs build/kv260_ooc
```

The script targets `xck26-sfvc784-2LV-c`, applies a provisional 250 MHz clock,
and emits utilization, timing, DRC, and synthesized-checkpoint artifacts. This
command has not run in the current macOS environment. A passing out-of-context
synthesis is not yet board proof; PS DDR integration, implementation, bitstream
generation, and measured output remain required.
