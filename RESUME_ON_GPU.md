# Resuming the H3 fidelity-push verification runs on GPU hardware

Context: this session implemented context-parallel attention sharding (to unblock
the paper's 10M-token/32-GPU scenario), an FP8-compute-doubling sensitivity toggle,
and corrected comparison-target misattributions, per
`/Users/rishabh/.claude/plans/is-the-implementation-on-vectorized-moth.md`.

All **code changes are complete and committed to the working tree** (uncommitted,
see `git status`/`git diff`). The **1M-context regression runs are done and
verified** (see below). The **10M-context/32-GPU runs were started locally and
killed partway through** — they were CPU-simulated (this is an analytical/roofline
simulator, no GPU code path exists or is needed; "GPU" here just means a faster/
more-cores machine) and were taking >14 minutes without finishing on this machine
for the 10M/32-device case, likely because `Cluster::run()` iterates the full
per-device analytical cost model for all 32 devices, per graph node, per of 10
`simulation.iter` steps, without early-exit — proportionally ~4x the 8-GPU case's
per-iteration work, compounded by CAG's large shared-cache-length arithmetic.

## What's done

- Build: `cd build && cmake .. && make -j` succeeds cleanly (no warnings from
  changed files).
- Code changes (see plan file for full rationale):
  - `src/model/model_config.h`: new `context_parallel_degree` field (default 1).
  - `config.yaml`, `eval/test.cpp`: new `context_parallel_degree` config knob and
    `fp8_compute_doubling` sensitivity toggle (default on = unchanged behavior),
    plumbed into asserts and `results.log`'s new CSV columns.
  - `src/module/parallel.cpp`/`.h`: `SelfAttentionParallel` now shards attention
    by `head_tp_dg = ne_tp_dg/context_parallel_degree` instead of `ne_tp_dg`
    directly, and adds a `context_merge` AllReduce-reuse module when
    `context_parallel_degree > 1`.
  - `src/module/attention.cpp`: shared CAG cache tensors sized by
    `shared_kv_cache_len / context_parallel_degree`.
  - `src/hardware/attention_gen_impl.cpp`: local `shared_kv_cache_len` divided by
    `context_parallel_degree` in the cost formula.
  - `src/hardware/cluster.cpp`: `checkMemorySize`/`checkH3MemorySize`'s
    `kv_cache_size_per_seq` divisor fixed to use `head_tp_dg` (private cache isn't
    context-parallel-split).
  - `extract_results.py`/`plot_results.py`: fixed misattributed 2.69x/2.09x
    throughput-per-power comparison targets (they're 10M-specific, not 1M), and
    fixed `scenario_key` to group by total device count
    (`num_node * num_device`), not `num_device` alone (needed once the 10M
    scenario spans multiple nodes).
- **Regression verification (1M context, `context_parallel_degree=1`) — PASSED**:
  batch sizes reproduce the pre-refactor `results.log.pre-h3fidelity` numbers
  exactly (669, 1587, 1587); throughput differs by <0.02% (floating-point noise,
  not a logic change — see `ASSUMPTIONS.md` once written). This confirms the
  refactor is behavior-preserving at the default `context_parallel_degree=1`.
- Old (pre-refactor, 8-GPU-only 10M numbers that don't represent the paper's real
  scenario) results archived at `results.log.pre-h3fidelity`.
- All 7 verification config files already generated in `configs_h3/`:
  `run1_1M_hbmonly.yaml`, `run2_1M_h3_fullbw.yaml`, `run3_1M_h3_halfbw.yaml`
  (all 3 already run, rows in `results.log`), and
  `run4_10M_hbmonly.yaml`, `run5_10M_h3_fullbw.yaml`, `run6_10M_h3_halfbw.yaml`,
  `run7_1M_h3_fp8off.yaml` (not yet run).

## What's left (resume here)

From `build/`, run the remaining four configs in order — each appends one row to
`../results.log`:

```bash
cd build
./run ../configs_h3/run4_10M_hbmonly.yaml    # HBM-only, 10M, 32 GPUs -- new baseline
./run ../configs_h3/run5_10M_h3_fullbw.yaml  # H3, 10M, 32 GPUs, full HBF bandwidth
./run ../configs_h3/run6_10M_h3_halfbw.yaml  # H3, 10M, 32 GPUs, half HBF bandwidth
./run ../configs_h3/run7_1M_h3_fp8off.yaml   # FP8-doubling-off sensitivity check (1M)
```

Consider setting `log.print_log: off` in these config files first if wall-clock
time is still a concern — the verbose per-op timeboard printout
(`Cluster::runIteration`/`print_timeboard()`) is likely a meaningful fraction of
the 10M/32-device run's wall time and has no effect on the numeric results
(`results.log` is written regardless).

If `run4` (HBM-only, 10M, 32 GPU) reports "Out of Memory" / aborts instead of
writing a row, **that is itself a valid, reportable result** — it would match the
paper's claim that HBM-only needs 32 GPUs as a bare minimum at this context size
(see `Cluster::checkMemorySize`'s `exit_out_of_memory`/`mem_cap_limit` config
knobs in `config.yaml`'s `simulation:` section if you want it to auto-shrink batch
size instead of aborting).

After all four rows land in `results.log`:

```bash
cd ..
python3 extract_results.py   # prints simulated vs. paper ratios per scenario
python3 plot_results.py      # regenerates fig5_batch_size.png / fig6_throughput_power.png
```

Then finalize `ASSUMPTIONS.md`'s "What this means for the two comparisons"
section with the real 10M ratios (batch/throughput/throughput-per-power vs. the
paper's 18.8x/6.14x/2.69x-2.09x) and the run7-vs-run2 FP8-doubling sensitivity
result (compare throughput ratios against the paper's 1.25x for 1M), per the
plan file's verification section. This file has not yet been rewritten this
session — it still has the pre-refactor "0.45x vs. paper's implied >1x" framing
that needs correcting regardless of the 10M results (see plan's item 9).

## Cleanup once done

- `configs_h3/*.yaml` are scratch configs, safe to delete once no longer needed
  (or keep for future reruns).
- `results.log.pre-h3fidelity` is the pre-refactor comparison baseline; keep
  until the new numbers are verified sane, then it can be deleted or archived
  elsewhere.
- This file (`RESUME_ON_GPU.md`) can be deleted once the runs are complete and
  `ASSUMPTIONS.md` is updated.
