#!/usr/bin/env python3
"""
H3 Action Item 2 -- HBF endurance under retention-driven refresh.

BACKGROUND
    H3 places only read-only data in HBF (model weights + the shared CAG KV
    cache) and argues that low HBF write endurance is therefore a non-issue.
    Measurement confirms the capacity-spill path never fires: private KV is
    ~123 MiB per unit of batch per device against a 192 GiB HBM budget, so
    steady-state HBF write traffic is exactly zero.

    But both FlashAccel and "Exploring HBF" propose relaxing NAND retention
    (3 years -> ~3 days) to buy 10-50x more P/E cycles. For data that is
    genuinely read-only, relaxed retention is not free: the resident data
    must be REWRITTEN every retention period or it is lost. That converts
    H3's read-only HBF contents into a periodic write stream.

    This script asks whether that trade is ever favourable.

MODEL
    Let V   = static bytes resident in HBF per device (weights + shared KV)
        T   = retention period
        T0  = baseline retention (3 years) at which P/E = pe_base

    Refresh write rate:      R(T) = V * WAF / T                    (~ 1/T)
    Endurance scaling:       PE(T) = pe_base * (T0/T)^alpha
    Write budget:            TBW(T) = V_usable * PE(T)
    Lifetime:                L(T) = TBW(T) / R(T)  ~  T^(1-alpha)

    alpha is calibrated from the published endurance gains at T = 3 days:
        10x  (conservative, Exploring HBF)  -> alpha = ln(10)/ln(365) = 0.390
        50x  (aggressive,  FlashAccel)      -> alpha = ln(50)/ln(365) = 0.663

    Since alpha < 1 in both cases, L(T) is MONOTONE INCREASING in T: the
    refresh traffic that relaxation forces costs more endurance than the
    relaxation buys. There is no interior optimum.
"""

import math

GiB = 2**30
YEAR = 365 * 24 * 3600
DAY = 24 * 3600


class HbfConfig:
    """Physical + deployment parameters. Defaults from the HBF literature."""

    def __init__(self,
                 hbf_capacity_gib=512.0,
                 static_resident_gib=20.4,   # weights + shared KV, per device
                 pe_base=100_000,            # SLC at 3-year retention
                 waf=1.02,                   # append-only, request-local
                 read_bw=1.6e12,             # per stack
                 t_r=3e-6, t_prog=100e-6,    # Z-NAND
                 baseline_retention_s=3 * YEAR):
        self.hbf_capacity = hbf_capacity_gib * GiB
        self.static_resident = static_resident_gib * GiB
        self.pe_base = pe_base
        self.waf = waf
        self.read_bw = read_bw
        self.t_r, self.t_prog = t_r, t_prog
        self.T0 = baseline_retention_s

    @property
    def write_bw(self):
        return self.read_bw * (self.t_r / self.t_prog)

    @property
    def usable(self):
        return self.hbf_capacity - self.static_resident


def alpha_from_gain(gain, T=3 * DAY, T0=3 * YEAR):
    """Calibrate the endurance exponent from a published gain at retention T."""
    return math.log(gain) / math.log(T0 / T)


def pe_effective(cfg, T, alpha):
    return cfg.pe_base * (cfg.T0 / T) ** alpha


def refresh_rate(cfg, T):
    """Bytes/s of media writes forced by rewriting resident data every T."""
    return cfg.static_resident * cfg.waf / T


def lifetime_years(cfg, T, alpha):
    r = refresh_rate(cfg, T)
    if r <= 0:
        return float('inf')
    return (cfg.usable * pe_effective(cfg, T, alpha)) / r / YEAR


def fmt_T(T):
    if T >= YEAR:
        return f"{T/YEAR:.2g} yr"
    if T >= DAY:
        return f"{T/DAY:.3g} d"
    return f"{T/3600:.3g} h"


def sweep(cfg, gain, label):
    alpha = alpha_from_gain(gain)
    print(f"\n{'='*74}")
    print(f"  RETENTION SWEEP  --  {label} (endurance gain {gain}x at 3 days, "
          f"alpha={alpha:.3f})")
    print(f"{'='*74}")
    print(f"{'retention':>11} {'P/E':>11} {'refresh rate':>14} "
          f"{'TBW':>10} {'lifetime':>12}")
    print("-" * 74)
    for T in [3*DAY, 7*DAY, 30*DAY, 90*DAY, 180*DAY,
              1*YEAR, 3*YEAR]:
        pe = pe_effective(cfg, T, alpha)
        r = refresh_rate(cfg, T)
        tbw = cfg.usable * pe
        L = lifetime_years(cfg, T, alpha)
        print(f"{fmt_T(T):>11} {pe:11,.0f} {r/1e6:11.1f} MB/s "
              f"{tbw/1e15:7.1f} PB {L:9.2f} yr")


def budget_analysis(cfg, target_years=5):
    print(f"\n{'='*74}")
    print(f"  WRITE BUDGET  --  what a {target_years}-year deployment allows")
    print(f"{'='*74}")
    tbw = cfg.usable * cfg.pe_base
    allowed = tbw / (target_years * YEAR)
    print(f"  usable HBF                : {cfg.usable/GiB:,.0f} GiB")
    print(f"  TBW at {cfg.pe_base:,} P/E     : {tbw/1e15:.1f} PB")
    print(f"  sustained write allowance : {allowed/1e6:.1f} MB/s per device")
    print(f"  HBF write bandwidth       : {cfg.write_bw/1e9:,.0f} GB/s")
    print(f"  -> allowance is {allowed/cfg.write_bw*100:.4f}% of write bandwidth")
    print()
    print(f"  Floor case (writes saturate HBF write bandwidth continuously):")
    floor = tbw / (cfg.write_bw * cfg.waf)
    print(f"      lifetime = {floor/DAY:.1f} days")
    print()
    print(f"  Refresh traffic vs allowance:")
    for T, name in [(3*DAY, "3-day retention"), (30*DAY, "30-day"),
                    (1*YEAR, "1-year"), (3*YEAR, "3-year (baseline)")]:
        r = refresh_rate(cfg, T)
        verdict = "OK" if r <= allowed else f"OVER by {r/allowed:.1f}x"
        print(f"      {name:20s} {r/1e6:9.1f} MB/s   {verdict}")


def corpus_update_analysis(cfg, target_years=5):
    """Case 1: corpus changes and must be re-prefilled, independent of retention."""
    print(f"\n{'='*74}")
    print(f"  CORPUS REFRESH  --  how often can the shared cache be updated?")
    print(f"{'='*74}")
    tbw = cfg.usable * cfg.pe_base
    allowed = tbw / (target_years * YEAR)
    shared_only = cfg.static_resident  # upper bound; weights rarely change
    min_period = shared_only * cfg.waf / allowed
    print(f"  resident bytes rewritten per update : {shared_only/GiB:,.1f} GiB")
    print(f"  minimum update period for {target_years} yr    : "
          f"{min_period/3600:.1f} h ({min_period/DAY:.2f} d)")
    print()
    print(f"{'update period':>15} {'write rate':>13} {'lifetime':>12}")
    print("-" * 45)
    for T in [1*3600, 6*3600, 1*DAY, 7*DAY, 30*DAY, 365*DAY]:
        r = shared_only * cfg.waf / T
        L = tbw / r / YEAR
        print(f"{fmt_T(T):>15} {r/1e6:10.1f} MB/s {L:9.1f} yr")


if __name__ == "__main__":
    cfg = HbfConfig()
    print("H3 HBF ENDURANCE MODEL")
    print(f"  HBF capacity      : {cfg.hbf_capacity/GiB:,.0f} GiB/device")
    print(f"  static resident   : {cfg.static_resident/GiB:,.1f} GiB "
          f"(weights + shared KV)")
    print(f"  usable            : {cfg.usable/GiB:,.0f} GiB")
    print(f"  write bandwidth   : {cfg.write_bw/1e9:,.0f} GB/s "
          f"(= read x tR/tPROG)")

    budget_analysis(cfg)
    sweep(cfg, 10, "conservative")
    sweep(cfg, 50, "aggressive")
    corpus_update_analysis(cfg)

    print(f"\n{'='*74}")
    print("  CONCLUSION")
    print(f"{'='*74}")
    print("""  Lifetime scales as T^(1-alpha) with alpha < 1 under both published
  endurance-gain figures, so it is monotone increasing in retention:
  relaxing retention is strictly counterproductive for read-only data.
  The refresh writes it forces consume endurance faster than the extra
  P/E cycles supply it.

  Retention relaxation pays only for data that is being rewritten anyway
  at a period shorter than the relaxed retention -- i.e. session KV
  write-back or a frequently-updated corpus, NOT H3's static shared cache.""")