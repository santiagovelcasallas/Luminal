"""HANDOVER §4.6: per-matrix sensitivity on calibration data + greedy knapsack."""
import time


def sensitivity(bench, store, names, opt_names, ref_opt, sens_ids, log=print):
    """Δ(m,o) = log ppl(sens_ids | only m in o, rest in ref_opt) − log ppl(sens_ids | all in ref_opt)."""
    bench.load({m: store[m][ref_opt].weff(bench.device) for m in names})
    base = bench.logppl(sens_ids)
    D = {m: {ref_opt: 0.0} for m in names}
    t0 = time.time()
    for j, m in enumerate(names):
        for o in opt_names:
            if o == ref_opt:
                continue
            bench.set_one(m, store[m][o].weff(bench.device))
            D[m][o] = bench.logppl(sens_ids) - base
        bench.set_one(m, store[m][ref_opt].weff(bench.device))
        if (j + 1) % 28 == 0:
            log(f"    sensitivity {j + 1}/{len(names)} ({time.time() - t0:.0f}s)")
    return base, D


def knapsack(names, opt_names, bits, delta, budget):
    """Start from the cheapest option per matrix; repeatedly apply the upgrade (m: o→o') with the largest
    (Δ(m,o) − Δ(m,o')) / (bits(o') − bits(o)) that keeps the total within budget. Only upgrades that lower Δ
    are considered. Returns None if even the cheapest assignment exceeds the budget (infeasible)."""
    choice = {m: min(opt_names, key=lambda o: bits[m][o]) for m in names}
    total = sum(bits[m][choice[m]] for m in names)
    if total > budget:
        return None, total
    while True:
        best = None
        for m in names:
            o = choice[m]
            for o2 in opt_names:
                db = bits[m][o2] - bits[m][o]
                gain = delta[m][o] - delta[m][o2]
                if db <= 0 or gain <= 0 or total + db > budget:
                    continue
                r = gain / db
                if best is None or r > best[0]:
                    best = (r, m, o2, db)
        if best is None:
            return choice, total
        _, m, o2, db = best
        choice[m] = o2
        total += db
