# Benchmark profile runbook

This machine's benchmark profile dedicates Ryzen 9 9950X CCD1 to benchmark
work:

- CPUs `8-15` are the CCD1 primary hardware threads used for benchmarks.
- CPUs `24-31` are their SMT siblings and are taken offline.
- CCD0 (`0-7,16-23`) remains available for the OS.

The benchmark Limine entries add:

```text
isolcpus=8-15,24-31 nohz_full=8-15,24-31 rcu_nocbs=8-15,24-31
```

The `bench-prep.service` service then takes CPUs `24-31` offline and sets the
CPU frequency governor for CPUs `8-15` to `performance`.

## Before running a benchmark

Boot a current `linux-cachyos benchmark` entry directly beneath the CachyOS
menu. Do not select one of the historical benchmark entries inside the
Snapshots submenu, because those entries do not contain the current isolation
parameters.

After logging in, check the preparation service and CPU state:

```bash
systemctl is-active bench-prep.service
cat /sys/devices/system/cpu/isolated
cat /sys/devices/system/cpu/offline
```

The expected output is:

```text
active
8-15,24-31
24-31
```

The complete boot command line can also be checked with:

```bash
cat /proc/cmdline
```

It should contain `isolcpus=8-15,24-31`, `nohz_full=8-15,24-31`, and
`rcu_nocbs=8-15,24-31` exactly once each.

## Pinning the benchmark

CPU isolation prevents ordinary scheduler work from being assigned to the
benchmark CPUs, but it does not automatically assign the benchmark to them.
Run `just` through `taskset`; Cargo, Criterion, and the benchmark executable
will inherit its affinity.

For the current single-threaded Criterion benchmarks, pinning to one physical
core prevents migration and is the most repeatable option:

```bash
taskset -c 8 just bench-v6-eytzinger-focus baseline
```

Here, `baseline` is the benchmark result key. To generate the default UTC
timestamp result key instead, omit it:

```bash
taskset -c 8 just bench-v6-eytzinger-focus
```

For a benchmark that intentionally needs multiple cores, allow all eight
isolated CCD1 physical cores:

```bash
taskset -c 8-15 just bench-v6-eytzinger-focus baseline
```

The `8-15` form confines the process tree to CCD1, but permits migration among
those cores. Prefer the single-CPU form for a single-threaded benchmark.

## Machine setup maintenance

The persistent Limine hook and benchmark entries are installed or refreshed
with:

```bash
sudo ~/restore-benchmark-boot.sh
```

The CPU preparation executable and systemd service are installed with:

```bash
sudo ~/install-bench-prep.sh
```

The service is conditioned on the benchmark kernel command line and is skipped
on normal boots. Its executable also verifies all three isolation parameters
and the expected CCD1 SMT topology before changing CPU state.
