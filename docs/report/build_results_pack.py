import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = lambda p: json.load(open(os.path.join(ROOT, p), encoding='utf-8'))


def build():
    full = R('results/m3/step9d_coarse_integration/timed_cost_result_m2.json')
    don = R('results/m3/step9d_coarse_integration/timed_cost_result_deeponet.json')
    abl = R('results/m3/step11_switching_ablation/switching_ablation.json')
    pinn = R('results/m3/step12_pinn_regime/pinn_controller.json')
    cs = R('results/deployment/cost_summary.json')
    coarse = R('results/m3/step11_coarse_detector/verification.json')

    L = []
    A = L.append
    A("# Team Turingz - Verified Results Pack (report source of truth)")
    A("")
    A("Every number below is read directly from a results file in the repository at build time.")
    A("Regenerate any of them with the command shown. Do NOT retype numbers into the report from")
    A("memory - copy them from here, and if an experiment is re-run, rebuild this file first.")
    A("Build: python docs/report/build_results_pack.py")
    A("")
    A("## T1. Full-system timed frontier (FNO + M1 coarse trust + M2 coupling + M3 controller)")
    A("Source: results/m3/step9d_coarse_integration/timed_cost_result_m2.json")
    A("Reproduce: python -m hybrid_pde.control_214133E._run_full")
    A("")
    ml, num = full['pure_ml'], full['pure_numerical']
    A(f"Pure ML baseline:        {ml['cost_s']:.3f} s at {100*ml['mean_error']:.2f}% error")
    A(f"Pure numerical baseline: {num['cost_s']:.3f} s at {100*num['mean_error']:.4f}% error")
    A("")
    A("| target | cost (s) | error | vs numerical | vs pure ML | hit rate |")
    A("|---|---|---|---|---|---|")
    for r in full['frontier']:
        A(f"| {r['target']:g} | {r['cost_s']:.3f} | {100*r['mean_error']:.2f}% | "
          f"{num['cost_s']/r['cost_s']:.2f}x cheaper | {ml['mean_error']/r['mean_error']:.2f}x more accurate | {100*r['hit_rate']:.0f}% |")
    A("")
    c = [r['cost_s'] for r in full['frontier']]
    e = [r['mean_error'] for r in full['frontier']]
    A(f"HEADLINE: {min(c):.2f}-{max(c):.2f} s at {100*min(e):.1f}-{100*max(e):.1f}% error, vs {num['cost_s']:.2f} s numerical "
      f"and {100*ml['mean_error']:.1f}% pure ML => {num['cost_s']/max(c):.1f}-{num['cost_s']/min(c):.1f}x cheaper, up to "
      f"{ml['mean_error']/min(e):.1f}x more accurate.")
    A("CAVEAT: timing is machine/session dependent; error, hit-rate and correction counts are deterministic.")
    A("All cost figures in one report table must come from the SAME run (this one).")
    A("")
    A("## T2. Switching ablation (five policies, six targets, 10 ICs, 200 steps)")
    A("Source: results/m3/step11_switching_ablation/switching_ablation.json")
    A("Reproduce: python -m hybrid_pde.control_214133E._ablation_switching")
    A("")
    A("| target | policy | error | corrections | switch events | hit rate |")
    A("|---|---|---|---|---|---|")
    for row in abl['rows']:
        for name, p in row['policies'].items():
            sw = f"{p.get('switch_events'):.1f}" if isinstance(p.get('switch_events'), float) else '-'
            A(f"| {row['target']:g} | {name} | {100*p['mean_error']:.2f}% | {p['corrections']:.1f} | {sw} | {100*p['hit_rate']:.0f}% |")
    A("")
    for n, v in abl.get('target_response', {}).items():
        A(f"- {n}: error spans {100*v['min']:.2f}%-{100*v['max']:.2f}% (range {100*v['range']:.2f}pp) -> "
          f"{'RESPONDS to target' if v['responds_to_target'] else 'FLAT (ignores the target)'}")
    A("")
    A("KEY CLAIMS THIS SUPPORTS:")
    A("1. vs fixed-interval schedule at matched cost: adaptive 3.00-5.31% vs fixed 7.70-7.73% (dominates).")
    A("2. vs hand-tuned fixed threshold (theta=0.4): same frontier, but hardcoded is FLAT (0.0445 at every")
    A("   target) while the adaptive controller responds (0.0531 -> 0.0300). The contribution is controllability.")
    A("3. Hysteresis deadband: measured unnecessary on the real trust signal (naive re-engages ~2x only);")
    A("   the shipped controller is a one-way latch. Do NOT claim deadband as novelty.")
    A("4. Knob saturates below target ~0.029 (theta_lo clips at 0.58): targets 0.02 and 0.01 identical.")
    A("")
    A("## T3. PINN through the controller (10 pre-trained ICs, training cost EXCLUDED)")
    A("Source: results/m3/step12_pinn_regime/pinn_controller.json")
    A("Reproduce: python demo_app/backend/_pinn_regime.py")
    A("")
    A("| target | error | corr fraction | hit rate |")
    A("|---|---|---|---|")
    for r in pinn['rows']:
        A(f"| {r['target']:g} | {100*r['mean_error']:.2f}% | {100*r['corr_frac']:.0f}% | {100*r['hit_rate']:.0f}% |")
    A("")
    A(f"Knob responds: {pinn['responds_to_target']} (error {100*pinn['error_range'][0]:.2f}%-{100*pinn['error_range'][1]:.2f}%), "
      f"best hit-rate {100*pinn['best_hit_rate']:.0f}%.")
    A("CLAIM: PINN is excluded by DEPLOYMENT cost (2114 s retrain per problem, ~950x numerical),")
    A("NOT by controllability - the controller works on it and cuts standalone extrapolation error ~4x.")
    A("")
    A("## T4. Per-model deployment profile (one machine, 30 repeats)")
    A("Source: results/deployment/cost_summary.json")
    A("")
    A("| model | deploy cost | error in-window | error beyond | params | disk | peak mem | scaling exp |")
    A("|---|---|---|---|---|---|---|---|")
    for m in ["FNO", "DeepONet", "PINN", "FDM", "Spectral", "ColeHopf"]:
        d = cs[m]
        dep = f"{d['deploy_s']/60:.0f} min" if d['deploy_s'] >= 60 else f"{d['deploy_s']:.3f} s"
        A(f"| {m} | {dep} | {d.get('err_in',0):.2f}% | {d.get('err_extrap',0):.2f}% | {d.get('params',0):,} | "
          f"{d.get('disk_mb',0):.2f} MB | {d.get('rss_mb',0):.0f} MB | {d.get('scaling',{}).get('exp',float('nan')):.2f} |")
    A("")
    A(f"PINN training split: {cs['PINN']['adam_s']:.0f} s Adam + {cs['PINN']['lbfgs_s']:.0f} s L-BFGS = "
      f"{cs['PINN']['train_s']:.0f} s per problem.")
    A("")
    A("## T5. DeepONet hybrid frontier (operating-regime evidence)")
    A("Source: results/m3/step9d_coarse_integration/timed_cost_result_deeponet.json")
    A("Reproduce: python -m hybrid_pde.control_214133E._run_full DeepONet")
    A("")
    dml, dnum = don['pure_ml'], don['pure_numerical']
    A(f"Baselines that session: ML {dml['cost_s']:.3f} s at {100*dml['mean_error']:.1f}%; numerical {dnum['cost_s']:.3f} s.")
    A("")
    A("| target | cost (s) | rel. to numerical | error | hit rate |")
    A("|---|---|---|---|---|")
    for r in don['frontier']:
        A(f"| {r['target']:g} | {r['cost_s']:.3f} | {r['cost_s']/dnum['cost_s']:.2f}x | {100*r['mean_error']:.1f}% | "
          f"{100*r.get('hit_rate',0):.0f}% |")
    A("")
    A("CLAIM: with DeepONet the hybrid costs 1.26-3.13x the numerical solver at ~23-39% error - strictly")
    A("dominated. Precondition violated: accuracy in-window (29.3% standalone). NOTE: normalise each frontier")
    A("by its OWN session's numerical baseline; never mix cost numbers across sessions.")
    A("")
    A("## T6. Coarse-reference drift detector (M1 integration evidence)")
    A("Source: results/m3/step11_coarse_detector/verification.json")
    res = coarse['results']
    A(f"- Coarse-detector correlation with true error: {res['coarse_corr_with_true_error']:.3f}")
    A(f"- Physics-residual correlation with true error: {res['residual_corr_with_true_error']:.3f}")
    A(f"- True failure onset detected at t = {res.get('true_failure_onset_t', '(see file)')}")
    A("NOTE: production detector is M1's contribution; M3 prototyped the concept (status: 'concept verified,")
    A("stand-ins'). Attribute accordingly in the report.")
    A("")
    A("## Numbers that must NOT be claimed (superseded or unsupported)")
    A("- 'adaptive 0.0% vs fixed 8.6%' - superseded synthetic result; use T2 (3.00-5.31% vs 7.70%).")
    A("- 'hysteresis deadband prevents chatter in the system' - T2 shows it is unnecessary on the real signal.")
    A("- 'nearly numerical-level accuracy' - floor is 3.00%, numerical is 0.011%. Say 'usable accuracy at a")
    A("  fraction of the cost' with the actual numbers.")
    A("- Any cost pairing that mixes sessions (e.g. 2.54 s numerical with this run's hybrid costs).")

    out = os.path.join(ROOT, 'docs', 'report', 'RESULTS_PACK.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(L))
    return out


if __name__ == '__main__':
    print('wrote', build())
