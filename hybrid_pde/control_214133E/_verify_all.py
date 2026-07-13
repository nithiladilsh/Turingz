from __future__ import annotations
import csv, io, contextlib, importlib
import numpy as np
from . import config
from .groundtruth import relative_l2
from ._smoke import run as smoke_run, NumExact, MLDrift, HORIZON
from .profiler import Profiler
from .accuracy_cost import AccuracyCostModel, AccuracyCostPoint
from .coupling import CouplingStub
from .trigger import SyntheticTrust
from .controller import AdaptiveController, FixedIntervalController, thresholds_for_target
from .pareto import ParetoPoint, dominates
from .robustness import hit_rate, mean_std
from . import integrate
from .demo import default_standins, demo_frame
from .runtime import HybridRuntime

R = []
def check(name, cond): R.append((name, bool(cond)))


def main():
    ok = True
    for m in ["config","contracts","groundtruth","trigger","coupling","profiler",
              "accuracy_cost","controller","runtime","_smoke"]:
        try: importlib.import_module(f"hybrid_pde.control_214133E.{m}")
        except Exception as e: ok = False; print("IMPORT FAIL", m, e)
    check("all modules import", ok)

    csvp = config.DATA_DIR / "colehopf" / "burgers_colehopf.csv"
    x=[]; u=[]
    with open(csvp) as f:
        rd=csv.reader(f); next(rd)
        for a,b,c in rd: x.append(float(b)); u.append(float(c))
    x=np.array(x); u=np.array(u); nx=len(np.unique(x)); U=u[:200*nx].reshape(200,nx)
    check("S1 metric self=0", relative_l2(U,U)==0.0)
    check("S1 metric 0.9x=0.1", abs(relative_l2(0.9*U,U)-0.1)<1e-9)
    check("S1 metric zero=1.0", abs(relative_l2(np.zeros_like(U),U)-1.0)<1e-9)

    with contextlib.redirect_stdout(io.StringIO()):
        s2 = smoke_run()
    check("S2 smoke PASS", s2)

    prof = Profiler(repeats=15, warmup=3)
    check("S3 scaling_fit=1.5", abs(prof.scaling_fit([1e3,2e3,4e3,8e3],3.3*np.array([1e3,2e3,4e3,8e3])**1.5)-1.5)<1e-6)
    mn,sd,md = prof.timeit(lambda: sum(range(5000)), repeats=10, warmup=2)
    check("S3 timeit sane", md>0 and sd>=0)

    model = AccuracyCostModel(1.0,1.0)
    check("S4 additive cost", model.predict_cost(100,10)==110.0)
    model.points=[AccuracyCostPoint(e,err,e) for e,err in [(0,0.30),(50,0.10),(100,0.02),(150,0.0)]]
    check("S4 inversion", model.budget_to_effort(0.05)==100)

    rng=np.random.default_rng(0); n=200; t=np.linspace(0,2,n)
    trust=np.clip(1/(1+np.exp((t-1)/0.08))+0.12*rng.standard_normal(n),0,1)
    tog=lambda d: int(np.sum(d[1:]!=d[:-1]))
    ac=AdaptiveController(0.4,0.6); ac.reset()
    hy=np.array([ac.decide(float(trust[i]),False,t[i],i).correct for i in range(n)])
    check("S5 hysteresis<single toggles", tog(hy)<tog(trust<0.5))
    m2=AccuracyCostModel(1,1,[AccuracyCostPoint(e,err,e) for e,err in [(0,0.3),(50,0.1),(150,0.01)]])
    a2=AdaptiveController(model=m2); a2.configure(0.10); lo=a2._horizon; a2.configure(0.01); hi=a2._horizon
    check("S5 tighter target->more correction", hi>=lo)

    xx=np.linspace(-1,1,512); tt=np.linspace(0,2,200); ic=np.sin(np.pi*xx)
    truth=ic[None,:]*np.exp(-tt)[:,None]
    rt=HybridRuntime(MLDrift(),NumExact(),SyntheticTrust(HORIZON,width=0.05,flag_at=0.5),
                     CouplingStub(),AdaptiveController(0.4,0.6))
    res=rt.run(ic,xx,tt,0.05,reference=truth)
    check("S6 accounting exact", res.cost.ml_steps+res.cost.correction_steps==200)
    check("S6 hybrid beats pure-ML", res.cost.achieved_error < relative_l2(MLDrift().rollout(ic,xx,tt),truth))
    check("S6 met target", res.cost.met_target)

    lo,hi=thresholds_for_target(0.02); check("S7 tighter target higher theta_lo", thresholds_for_target(0.02)[0] > thresholds_for_target(0.30)[0])
    check("S7 dominance fn", dominates(ParetoPoint("a",1,1),ParetoPoint("b",2,2)) and not dominates(ParetoPoint("a",1,2),ParetoPoint("b",2,1)))
    def _pt(tg):
        l,h=thresholds_for_target(tg)
        r=HybridRuntime(MLDrift(),NumExact(),SyntheticTrust(HORIZON,width=0.08,flag_at=0.0),CouplingStub(),AdaptiveController(l,h)).run(ic,xx,tt,tg,reference=truth)
        return r.cost.ml_steps*1.0+r.cost.correction_steps*10.0, r.cost.achieved_error
    pts=[_pt(tg) for tg in [0.30,0.10,0.02]]
    check("S7 knob traces trade-off", pts[0][1]>=pts[2][1]-1e-9 and pts[0][0]<=pts[2][0]+1e-9)
    check("S7 numerical-grade accuracy cheaper", min(p[1] for p in pts)<=1e-3 and min(p[0] for p in pts)<200*10.0)
    check("S8 hit_rate helper", hit_rate([0.01,0.2,0.03],0.05)==2/3 and mean_std([1.0,3.0])[0]==2.0)
    class _MLD:
        def __init__(self,ss): self.s=ss
        def rollout(self,ic,xx_,tt_):
            tt_=np.asarray(tt_,float); ic=np.asarray(ic,float)
            val=np.where(tt_<=HORIZON,np.exp(-tt_),np.exp(-HORIZON)*(1-self.s*(tt_-HORIZON)))
            return ic[None,:]*val[:,None]
    _ics=[(np.sin(k*np.pi*xx),0.4+0.25*k) for k in (1,2,3,4)]
    def _err(mk,target):
        out=[]
        for ic,sstr in _ics:
            r=HybridRuntime(_MLD(sstr),NumExact(),SyntheticTrust(HORIZON,width=0.08,flag_at=0.0),CouplingStub(),mk(target)).run(ic,xx,tt,target,reference=ic[None,:]*np.exp(-tt)[:,None])
            out.append(r.cost.achieved_error)
        return out
    hra=hit_rate(_err(lambda t_:AdaptiveController(*thresholds_for_target(t_)),0.02),0.02)
    hrf=hit_rate(_err(lambda t_:FixedIntervalController(k=2),0.02),0.02)
    check("S8 adaptive more reliable than fixed (OOD)", hra>=hrf)
    _rows=integrate.dry_run()
    check("S9 integration engine dry-run", all("mean_error" in r for r in _rows) and all(0.0<=r["hit_rate"]<=1.0 for r in _rows))
    _sp=integrate.load_numerical_solver().rollout(np.sin(np.pi*xx), xx, tt)
    check("S9 spectral adapter works torch-free", _sp.shape==(200,512) and np.isfinite(_sp).all())
    _pending=0
    for _n in ["load_trust","load_coupling"]:
        try: getattr(integrate,_n)()
        except NotImplementedError: _pending+=1
    check("S9 M1/M2 hooks pending (raise with note)", _pending==2)
    _p=default_standins(); _fr=demo_frame(0.10,**_p)
    check("S10 demo backend frame complete", {"x","t","truth","ml","num","hybrid","trust","switch","cost","comparison"}.issubset(_fr) and _fr["hybrid"].shape==(200,512))
    _lo=demo_frame(0.30,**_p); _ti=demo_frame(0.01,**_p)
    check("S10 knob changes outcome", _ti["cost"].correction_steps>=_lo["cost"].correction_steps and _ti["cost"].achieved_error<=_lo["cost"].achieved_error)
    print("="*50)
    allok=True
    for name,c in R:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}"); allok=allok and c
    print("="*50)
    print("OVERALL:", "ALL PASS" if allok else "SOME FAILED")
    print("note: load_reference() (torch) verified separately in the project env")
    return allok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
