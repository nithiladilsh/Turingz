import { Card } from "../components/ui.jsx";

export default function Overview({ go }) {
  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-800">Reliability & Trust Estimation for a Hybrid PDE Solver</h1>
      <p className="text-slate-600 mt-2 max-w-3xl">
        Our hybrid solver runs a fast machine-learning model while it is trustworthy, then switches to an
        accurate numerical solver when needed. My module is the safety switch: it produces a live trust
        score with no access to the true answer and decides when to hand over.
      </p>

      <div className="mt-6 grid grid-cols-3 gap-4">
        <Card title="The problem">
          <p className="text-sm text-slate-600">ML solvers are fast but drift wrong over time. Numerical solvers are accurate but slow. We want the speed of one and the safety of the other.</p>
        </Card>
        <Card title="My contribution">
          <p className="text-sm text-slate-600">A reference-free trust score from physics signals, plus a cheap-reference mode for the hardest model (FNO). Outputs one number and a switch flag.</p>
        </Card>
        <Card title="The result">
          <p className="text-sm text-slate-600">On 100 unseen waves the switch is safe 95% of the time and never dangerously late; it also catches unfamiliar (out-of-distribution) inputs.</p>
        </Card>
      </div>

      <Card title="How the pieces connect" className="mt-4">
        <pre className="text-xs text-slate-600 leading-5 overflow-x-auto">{`
   initial wave  ->  [ ML model (fast) ]  ->  prediction u(x,t)
                                                   |
                                                   v
                                    [ MY MODULE: trust estimator ]
                                     physics signals  or  cheap reference
                                     -> trust score 0..1  +  switch flag
                                                   |
                              trust high? keep ML  |  trust low? switch to
                                                      numerical solver (accurate)
`}</pre>
        <button onClick={() => go("trust")}
          className="mt-3 px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700">
          Try the live Trust Score demo →
        </button>
      </Card>
    </div>
  );
}
