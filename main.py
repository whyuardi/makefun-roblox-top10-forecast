# VARIANT v108-hl36fastminp06: v107 + MIN_P 0.006
# Generated from v61 (submitted) — main.py untouched.
# VARIANT v61-v60rho95: v60 + RHO 0.95
# From render2_base (per-place shrink). main.py untouched.
#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads hourly snapshots of the chart from /input (MAKEFUN_INPUT_DIR) and writes
/output/forecast.json (MAKEFUN_OUTPUT_DIR): for each of the top 10 places a
probability for each game named being at that place, plus the top-10 guess.

Method: rank-space Monte Carlo diffusion, calibrated on the chart's
own history (49 public snapshots, 2026-10-30 -> 2026-11-12).
  * Every game in the chart is a particle with a rank and a small rank
    velocity (EWMA of its recent rank changes, shrunk 5x toward 0: rank
    velocity barely persists, so the raw EWMA would double-count movement).
  * Each simulated hour: velocity mean-reverts (AR(1)) with noise, rank moves
    by velocity + a small band drift + noise (band-dependent volatility,
    estimated from the history itself); games exit with a band/tenure hazard.
  * Breakout surges: a game can start a directed climb to a top-10 target
    rank over 8-20h, with a per-hour probability calibrated on the archive's
    P(top10 at +24h) by (rank band, chart age); young games surge, games
    older than 7 days outside the top 10 never return (0/558 in the archive).
  * Brand-new "phantom" games enter at low ranks (~0.9/hour); ~5% of them
    surge to the top 10. Their occupancy becomes the forecast's unnamed mass
    (about 8-10% per place, matching the share of top-10 outcomes that were
    games not in the chart at forecast time).
  * The chart is re-sorted every simulated hour; occupancy frequencies over
    the 72-hour window, averaged over 2,000 simulations, ARE the place
    marginals (the log score is proper for marginals, so this is the right
    target). Frequencies are shrunk 20% toward a persistence prior, floored
    at 0.004, then written out.
Standard library only. Deterministic: fixed RNG seed and fixed simulation
count; no wall-clock dependence in the output.
"""
import json
import math
import os
import random
import time
from datetime import datetime

# ---------------------------------------------------------------- tunables
SEED = 20261113
VEL_HALF_LIFE_H = 36.0
VEL_CAP = 0.5               # cap |velocity| (ranks/hour)
VEL_SHRINK = 0.2            # shrink EWMA velocity toward 0 (see estimate())
RHO = 0.95
SIGMA_V = 0.10              # hourly velocity noise (ranks/hour)
DRIFT_TOP = 0.02            # ranks/hour drift for rank <= 30 (downward)
DRIFT_LOW = -0.03           # ranks/hour drift for rank > 30 (upward)
NEW_PER_HOUR = 1.5
NEW_RANK_LO, NEW_RANK_HI = 40.0, 95.0
PHANTOM_SURGE_P = 0.12
SURGE_T_MIN, SURGE_T_MAX = 6, 14   # hours for a surge to reach its target
EXIT_TENURE_DAYS = 14.0     # extra exit hazard beyond this chart tenure
EXIT_TENURE_EXTRA = 0.004   # per hour
OLD_OUT_HAZARD = 0.008      # extra exit hazard for age>7d games outside top 10


def surge_prob(rank, age_h):
    """Per-hour probability a game starts a breakout surge to the top 10.
    Calibrated on the archive: P(top10 at +24h) by (rank band, age bucket).
    Games older than 7 days outside the top 10 NEVER return (0/558)."""
    if rank <= 10 or age_h > 7 * 24:
        return 0.0
    base = (0.009 if rank <= 20 else
            0.0025 if rank <= 40 else
            0.0015 if rank <= 70 else 0.0005)
    age_f = 1.0 if age_h < 24 else (0.8 if age_h < 72 else 0.5)
    return base * age_f
SHRINK = 0.20               # blend toward persistence prior (tuned: 0.2 best on 3 windows)
SHRINK_PER_PLACE = [0.30, 0.25, 0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]  # per-place shrink (place 1 most persistent)
UNNAMED_MULT = 1.0         # multiplier on phantom share -> unnamed mass per place
MIN_P = 0.006
MAX_ROWS_PER_PLACE = 100
BATCH = 100                   # simulations per batch inside mc_marginals
# Simulation count is FIXED (max_sims=2500 in mc_marginals/forecast) so the
# output is deterministic; time_budget_s is only an emergency brake.


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_history(input_dir):
    snaps = []
    path = os.path.join(input_dir, "history.ndjson")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("ok", True):
                continue
            uc = d.get("up_and_coming") or []
            rows = []
            for r in uc:
                try:
                    rows.append((int(r["rank"]), int(r["universe_id"]),
                                 float(r.get("playing") or 0)))
                except (KeyError, TypeError, ValueError):
                    continue
            if not rows:
                continue
            rows.sort()
            snaps.append({"at": parse_time(d["at"]), "rows": rows,
                          "details": d.get("details")})
    snaps.sort(key=lambda s: s["at"])
    return snaps


def estimate(snaps):
    """Estimate velocities, band volatilities and exit hazards from history."""
    per_game = {}   # uid -> list of (t, rank)
    for s in snaps:
        for rank, uid, _playing in s["rows"]:
            per_game.setdefault(uid, []).append((s["at"], rank))
    # interval velocities + band stats + exits
    band_num = {"top": [], "mid": [], "low": []}
    band_den = {"top": 0.0, "mid": 0.0, "low": 0.0}
    exits = {"top": 0, "mid": 0, "low": 0}
    exposure = {"top": 0.0, "mid": 0.0, "low": 0.0}
    for i in range(1, len(snaps)):
        t0, t1 = snaps[i - 1]["at"], snaps[i]["at"]
        dt = max((t1 - t0) / 3600.0, 0.5)
        prev = {uid: rank for rank, uid, _ in snaps[i - 1]["rows"]}
        cur = {uid: rank for rank, uid, _ in snaps[i]["rows"]}
        for uid, r0 in prev.items():
            b = "top" if r0 <= 10 else ("mid" if r0 <= 30 else "low")
            exposure[b] += dt
            if uid in cur:
                band_num[b].append((cur[uid] - r0) / math.sqrt(dt))
            else:
                exits[b] += 1
    def sd(xs):
        if len(xs) < 5:
            return None
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    sigma = {}
    for b, default in (("top", 0.32), ("mid", 0.84), ("low", 1.46)):
        v = sd(band_num[b])
        sigma[b] = min(max(v if v is not None else default, 0.05), 3.0)
    hazard = {}
    for b in ("top", "mid", "low"):
        h = exits[b] / exposure[b] if exposure[b] > 0 else 0.0
        hazard[b] = min(max(h, 0.0), 0.05)
    # per-game velocity: EWMA of interval velocities (half-life VEL_HALF_LIFE_H),
    # shrunk hard toward 0. Empirically, rank velocity barely persists
    # (24h velocity vs next-24h velocity corr ~0.1; lag-1 interval autocorr
    # ~0.2), so most of an interval's dr is noise already counted in sigma.
    # Using the raw EWMA double-counts movement and over-extrapolates.
    vel = {}
    first_seen = {}
    last_rank = {}
    last_playing = {}
    for uid, obs in per_game.items():
        first_seen[uid] = obs[0][0]
        last_rank[uid] = obs[-1][1]
        v = 0.0
        wsum = 0.0
        for j in range(1, len(obs)):
            dt = max((obs[j][0] - obs[j - 1][0]) / 3600.0, 0.5)
            iv = (obs[j][1] - obs[j - 1][1]) / dt
            age_h = (obs[-1][0] - obs[j][0]) / 3600.0
            w = 0.5 ** (age_h / VEL_HALF_LIFE_H)
            v += w * iv
            wsum += w
        raw = v / wsum if wsum > 0 else 0.0
        vel[uid] = max(-VEL_CAP, min(VEL_CAP, VEL_SHRINK * raw))
    for s in snaps:
        for rank, uid, playing in s["rows"]:
            last_playing[uid] = playing
    return {"sigma": sigma, "hazard": hazard, "vel": vel,
            "first_seen": first_seen, "last_rank": last_rank}


def simulate_once(games, est, hours, rng, record_hours=None):
    """One Monte Carlo run. games: list of dicts (uid, rank, vel, first_seen).
    Returns occupancy: occ[(uid, place)] counts, phantom_occ[place] counts.
    If record_hours is a set of hour indices (1-based), also returns
    rec[hour] = list of uids in the top 10 at that hour.
    State per game: [uid, rank, vel, first_seen, is_phantom, surge_target,
    surge_vel] -- surge_target None when not surging."""
    _rec = {} if record_hours else None
    sigma, hazard = est["sigma"], est["hazard"]
    st = [[g["uid"], float(g["rank"]), g["vel"], g["first_seen"], False,
           None, 0.0] for g in games]
    occ = {}
    phantom_occ = {k: 0 for k in range(1, 11)}
    t_end = games[0]["t_end"] if games else 0.0
    for _h in range(hours):
        now = t_end + (_h + 1) * 3600.0
        # brand-new games enter
        n_new = 0
        lam = NEW_PER_HOUR
        L = math.exp(-lam)
        p = 1.0
        while True:
            p *= rng.random()
            if p <= L:
                break
            n_new += 1
        for _ in range(n_new):
            r0 = rng.uniform(NEW_RANK_LO, NEW_RANK_HI)
            if rng.random() < PHANTOM_SURGE_P:
                tgt = rng.uniform(1.0, 10.0)
                T = rng.uniform(SURGE_T_MIN, SURGE_T_MAX)
                st.append([None, r0, 0.0, now, True, tgt, -(r0 - tgt) / T])
            else:
                st.append([None, r0, rng.uniform(-1.0, 0.0), now, True,
                           None, 0.0])
        alive = []
        for g in st:
            _uid, r, v, fs, ph, tgt, svel = g
            age_h = (now - fs) / 3600.0
            b = "top" if r <= 10 else ("mid" if r <= 30 else "low")
            # exit hazard
            h = hazard[b]
            if age_h > EXIT_TENURE_DAYS * 24:
                h += EXIT_TENURE_EXTRA
            if age_h > 7 * 24 and r > 10:
                h += OLD_OUT_HAZARD
            if rng.random() < h:
                continue
            if tgt is not None:
                # surging: directed climb toward the target rank
                r = r + svel + rng.gauss(0.0, 0.15)
                if r <= tgt:
                    r = tgt
                    g[5] = None
                    g[2] = 0.0
            else:
                # maybe start a surge (only for known games; phantoms
                # already got their surge chance at entry)
                if not ph and rng.random() < surge_prob(r, age_h):
                    tgt = rng.uniform(1.0, 10.0)
                    T = rng.uniform(SURGE_T_MIN, SURGE_T_MAX)
                    g[5] = tgt
                    g[6] = -(r - tgt) / T
                    r = r + g[6] + rng.gauss(0.0, 0.15)
                else:
                    v = RHO * v + rng.gauss(0.0, SIGMA_V)
                    v = max(-12.0, min(12.0, v))
                    drift = DRIFT_TOP if r <= 30 else DRIFT_LOW
                    r = r + v + drift + rng.gauss(0.0, sigma[b])
            if r < 0.5:
                r = 0.5 + (0.5 - r) * 0.3
            if r > 120.0:
                r = 120.0
            g[1] = r
            alive.append(g)
        st = alive
        if not st:
            break
        st.sort(key=lambda g: (g[1], g[0] if g[0] is not None else 0))
        if _rec is not None and (_h + 1) in record_hours:
            _rec[_h + 1] = [g[0] for g in st[:10] if not g[4]]
        for k in range(1, min(10, len(st)) + 1):
            g = st[k - 1]
            if g[4]:
                phantom_occ[k] += 1
            else:
                key = (g[0], k)
                occ[key] = occ.get(key, 0) + 1
    if _rec is not None:
        return occ, phantom_occ, _rec
    return occ, phantom_occ


def mc_marginals(snaps, places=10, hours=72, max_sims=2000,
                 time_budget_s=280.0, min_sims=2000):
    """Run the Monte Carlo and return RAW marginals (no shrinkage, no
    rendering): {'mc': {place: {uid: p}}, 'phantom': {place: share},
    'chart': [uid...], 'sims': n}."""
    est = estimate(snaps)
    last = snaps[-1]
    t_end = last["at"]
    games = []
    for rank, uid, _playing in last["rows"]:
        games.append({"uid": uid, "rank": float(rank),
                      "vel": est["vel"].get(uid, 0.0),
                      "first_seen": est["first_seen"].get(uid, t_end),
                      "t_end": t_end})
    chart = [g["uid"] for g in games]
    rng = random.Random(SEED)
    occ_total = {}
    phantom_total = {k: 0 for k in range(1, places + 1)}
    sims = 0
    t_start = time.time()
    while True:
        for _ in range(BATCH):
            occ, ph = simulate_once(games, est, hours, rng)
            for key, c in occ.items():
                occ_total[key] = occ_total.get(key, 0) + c
            for k in range(1, places + 1):
                phantom_total[k] += ph[k]
            sims += 1
        if sims >= max_sims:
            break
        if sims >= min_sims and time.time() - t_start > time_budget_s:
            break
    denom = sims * hours
    mc = {k: {} for k in range(1, places + 1)}
    for (uid, k), c in occ_total.items():
        mc[k][uid] = c / denom
    phantom = {k: phantom_total[k] / denom for k in range(1, places + 1)}
    return {"mc": mc, "phantom": phantom, "chart": chart, "sims": sims}


def render(mcres, places=10, shrink=None, min_p=MIN_P,
           unnamed_floor=0.01):
    """Turn raw MC marginals into the forecast dict (probabilities + top10).
    Shrinkage blends the MC marginals toward a persistence prior; the
    phantom share becomes the unnamed mass per place."""
    mc, phantom, chart = mcres["mc"], mcres["phantom"], mcres["chart"]
    prior = {k: {} for k in range(1, places + 1)}
    for k in range(1, places + 1):
        prior[k][chart[k - 1]] = 0.5
        if k == 1:
            if len(chart) > 1:
                prior[k][chart[1]] = prior[k].get(chart[1], 0.0) + 0.3
        else:
            prior[k][chart[k - 2]] = prior[k].get(chart[k - 2], 0.0) + 0.15
            if k < len(chart):
                prior[k][chart[k]] = prior[k].get(chart[k], 0.0) + 0.15
    probs = {k: {} for k in range(1, places + 1)}
    for k in range(1, places + 1):
        if shrink is None:
            sk = SHRINK_PER_PLACE[k - 1]
        elif isinstance(shrink, (list, tuple)):
            sk = shrink[k - 1]
        else:
            sk = shrink
        for uid, m in mc[k].items():
            probs[k][uid] = (1.0 - sk) * m + sk * prior[k].get(uid, 0.0)
        # games the MC never put at this place but the prior names get a
        # small mass too (persistence of the current chart)
        for uid, pr in prior[k].items():
            if uid not in probs[k]:
                probs[k][uid] = sk * pr
    rows = []
    for k in range(1, places + 1):
        cand = sorted(probs[k].items(), key=lambda kv: -kv[1])
        unnamed_target = max(phantom[k] * UNNAMED_MULT, unnamed_floor)
        named = []
        s = 0.0
        for uid, p in cand:
            if len(named) >= MAX_ROWS_PER_PLACE:
                break
            if p < min_p:
                break
            named.append([uid, p])
            s += p
        if s > 0 and s > 1.0 - unnamed_target:
            scale = (1.0 - unnamed_target) / s
            named = [[uid, p * scale] for uid, p in named]
            s = 1.0 - unnamed_target
        for uid, p in named:
            rows.append({"universe_id": uid, "place": k, "p": round(p, 6)})
    used = set()
    top10 = []
    for k in range(1, places + 1):
        best, bestp = None, -1.0
        for uid, p in probs[k].items():
            if uid in used:
                continue
            if p > bestp:
                best, bestp = uid, p
        if best is None:
            for uid in chart:
                if uid not in used:
                    best = uid
                    break
        used.add(best)
        top10.append(best)
    return {"probabilities": rows, "top10": top10,
            "_sims": mcres["sims"],
            "_phantom_share": {str(k): round(v, 4)
                               for k, v in phantom.items()}}


def forecast(snaps, places=10, hours=72, max_sims=2000, time_budget_s=280.0,
             min_sims=2000):
    mcres = mc_marginals(snaps, places, hours, max_sims, time_budget_s,
                         min_sims)
    return render(mcres, places)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task = json.load(open(os.path.join(input_dir, "task.json")))
    places = int(task.get("places", 10))
    window = task.get("window", {}) or {}
    hours = int(window.get("slots", 72))
    snaps = load_history(input_dir)
    result = forecast(snaps, places=places, hours=hours)
    result.pop("_sims", None)
    result.pop("_phantom_share", None)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task.get("output", "forecast.json")),
              "w", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
