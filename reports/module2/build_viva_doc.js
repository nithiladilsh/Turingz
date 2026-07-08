const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, ImageRun, PageBreak
} = require("docx");

const FIG = "/sessions/loving-sweet-fermat/mnt/Turingz/results/module2/figures";
const OUT = "/sessions/loving-sweet-fermat/mnt/Turingz/reports/module2/VIVA_Guide_Module2.docx";

// ---------- helpers ----------
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children:[new TextRun(t)] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children:[new TextRun(t)] });
const H3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, children:[new TextRun(t)] });
function P(text, opts={}) {
  return new Paragraph({ spacing:{after:120}, children:[new TextRun({ text, ...opts })] });
}
// paragraph from runs: array of [text, {bold?, italics?}]
function PR(runs, opts={}) {
  return new Paragraph({ spacing:{after:120}, ...opts,
    children: runs.map(r => new TextRun({ text:r[0], ...(r[1]||{}) })) });
}
function quote(text) {
  return new Paragraph({ spacing:{after:120}, indent:{left:480},
    border:{ left:{ style:BorderStyle.SINGLE, size:18, color:"2E75B6", space:12 } },
    children:[new TextRun({ text, italics:true })] });
}
function bullet(text) {
  return new Paragraph({ bullet:{level:0}, spacing:{after:60}, children:[new TextRun(text)] });
}
const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top:border, bottom:border, left:border, right:border };
function cell(text, w, {head=false, align=AlignmentType.LEFT}={}) {
  return new TableCell({ borders, width:{size:w,type:WidthType.DXA},
    shading:{ fill: head?"D5E8F0":"FFFFFF", type:ShadingType.CLEAR },
    margins:{top:60,bottom:60,left:100,right:100},
    children:[new Paragraph({ alignment:align, children:[new TextRun({text, bold:head, size:19})] })] });
}
function table(colW, rows) {
  return new Table({ width:{size:colW.reduce((a,b)=>a+b,0),type:WidthType.DXA}, columnWidths:colW,
    rows: rows.map((r,ri)=> new TableRow({ children: r.map(c=>cell(String(c), colW[r.indexOf(c)]??colW[0], {head:ri===0})) })) });
}
// safer table builder (index-correct widths)
function tbl(colW, rows){
  return new Table({ width:{size:colW.reduce((a,b)=>a+b,0),type:WidthType.DXA}, columnWidths:colW,
    rows: rows.map((r,ri)=> new TableRow({ children: r.map((c,ci)=> cell(String(c), colW[ci], {head:ri===0, align: ci===0?AlignmentType.LEFT:AlignmentType.CENTER})) })) });
}
function figure(file, w=470){
  const h = Math.round(w/1.628);
  return new Paragraph({ alignment:AlignmentType.CENTER, spacing:{before:80,after:80},
    children:[new ImageRun({ type:"png", data:fs.readFileSync(`${FIG}/${file}`),
      transformation:{width:w,height:h},
      altText:{title:file,description:file,name:file} })] });
}
const spacer = () => new Paragraph({ children:[new TextRun("")] });

// ---------- content ----------
const kids = [];
kids.push(new Paragraph({ heading:HeadingLevel.TITLE, children:[new TextRun("Module 2 - Coupling: Viva Guide")] }));
kids.push(PR([["Student: ", {bold:true}],["Dharmapala R.D. (214050V)    "],["Module: ", {bold:true}],["Coupling (ML-to-numerical handoff)"]]));
kids.push(P("A plain-English record of everything done so far - what we did, why, the results, and how to explain it in the viva. Updated at the end of every phase.", {italics:true}));

// 0
kids.push(H1("0. One-paragraph summary (say this if asked \"what is your module?\")"));
kids.push(quote("The team has two kinds of solver for a fluid-flow equation. One is a machine-learning model (fast, but it becomes wrong when asked to predict far into the future). The other is a numerical solver (slow, but always accurate). My module is the bridge between them: I run the fast model early, then hand its answer over to the accurate solver to finish the job. I then measure when this handover helps and when it doesn't. My experiment shows the handover cuts the future-prediction error from about 14% down to about 1% when done early."));

// 1
kids.push(H1("1. The big picture, in simple words"));
kids.push(PR([["The equation. ",{bold:true}],["We study the 1-D viscous Burgers equation - a simple model of a wave in a slightly sticky fluid. The wave moves, steepens, and the stickiness (viscosity) smooths it out. We want the shape of the wave u(x,t) at every position x and time t."]]));
kids.push(P("Two ways to solve it:"));
kids.push(bullet("Numerical solvers (Cole-Hopf, pseudo-spectral, finite-difference): follow the real maths step by step. Very accurate, but slow."));
kids.push(bullet("Machine-learning solvers (FNO, DeepONet, PINN): learn the answer from many examples. Very fast, but they were only trained up to time t = 1. Past that (extrapolation) they drift away from the truth."));
kids.push(PR([["The trade-off we exploit. ",{bold:true}],["Fast-but-wrong vs slow-but-right. A hybrid uses the fast one while it can be trusted and the accurate one only when needed."]]));
kids.push(P("Key numbers (our own measurements):"));
kids.push(tbl([3120,3120,3120], [
  ["FNO error inside training (t<=1)","FNO error extrapolating (t>1)","Trustworthy until (reliable horizon)"],
  ["0.57%","14.1%","t = 1.46"],
]));

// 2
kids.push(H1("2. What exactly is my contribution"));
kids.push(P("Two teammates handle when to switch (a \"trust\" signal) and how much compute to spend (cost control). I build and study the actual handover between the ML model and the numerical solver."));
kids.push(PR([["Scope (important). ",{bold:true}],["My module does not decide when to switch - it receives the switch time as an input. It answers: given this supplied switch time, what happens to accuracy when I hand the FNO's wave to the numerical solver? The handoff is one-way (we do not switch back to the ML model)."]]));
kids.push(quote("Research question: How does an externally supplied switch time - and the quality of the FNO's wave at that moment - affect the accuracy of FNO-to-numerical continuation?"));

// 3
kids.push(H1("3. Small glossary"));
kids.push(tbl([2600,6760], [
  ["Word","Simple meaning"],
  ["Viscosity","How sticky/smoothing the fluid is. Ours is 1/(100*pi), about 0.0032 (only slightly sticky)."],
  ["Extrapolation","Predicting beyond the times the model was trained on (here, t > 1)."],
  ["Handoff","Taking the ML wave at a chosen time and giving it to the numerical solver as a fresh start."],
  ["Switch time (t_s)","The moment we hand over from ML to numerical."],
  ["Reference / ground truth","The correct answer. We use the Cole-Hopf solver."],
  ["Relative L2 error","One number for how wrong a prediction is (0 = perfect)."],
  ["Held-out waves","Starting waves the ML model never saw in training - a fair test."],
]));

// 4
kids.push(H1("4. The journey (key decisions)"));
kids.push(bullet("Use the pseudo-spectral solver as the continuer and keep Cole-Hopf as an independent referee - so no one can say we \"inserted the true answer\"."));
kids.push(bullet("Test on many starting waves, not one - a single example is an anecdote; 10-20 waves are evidence."));
kids.push(bullet("Add a perfect-handoff yardstick (restart from the true wave) to separate ML error from solver error."));
kids.push(bullet("Fix the success rule in advance (at least 10% improvement) so we don't fool ourselves."));

// 5
kids.push(H1("5. Phase 1 - The Audit"));
kids.push(PR([["What we did. ",{bold:true}],["Before writing hybrid code, we read the team's real code and wrote an exact \"contract\": grid size, viscosity, time steps, which model is real, and whether the numerical solver can restart."]]));
kids.push(PR([["Why. ",{bold:true}],["If two systems secretly disagree on a detail (viscosity, or whether numbers are normalised), the hybrid can look correct but be silently wrong."]]));
kids.push(P("What we found - three important things:"));
kids.push(PR([["1) The FNO is NOT a step-by-step model. ",{bold:true}],["We had assumed it predicts one small step at a time with errors piling up. The real model takes (starting wave, a time t, position x) and jumps directly to the answer at that time, in one shot - a direct \"time machine\", not a step-by-step walker. So the correct story is: it simply gets less accurate the further past t=1 you ask it."]]));
kids.push(PR([["2) The numerical solver could not restart. ",{bold:true}],["The team's pseudo-spectral solver only ran from the very beginning (t=0). To hand over in the middle, we had to teach it to start from an arbitrary wave."]]));
kids.push(PR([["3) The dataset was not on disk, ",{bold:true}],["but a results file already stored the true answers and the FNO's answers for 10 unseen test waves - enough to run real experiments immediately, without re-training."]]));
kids.push(quote("Say in the viva: In my audit I discovered our FNO is a direct space-time operator, not autoregressive - so I corrected the framing before writing code. I also found the numerical solver couldn't restart mid-run, which became my first engineering task."));

// 6
kids.push(H1("6. Phase 2 - Restart gate and first hybrid result"));
kids.push(PR([["The safety check (gate). ",{bold:true}],["Before trusting any hybrid result: if we restart the numerical solver from the perfectly correct wave, does it reproduce the correct future? Yes - the error was about 0.000005 (essentially zero). So the restart machinery is correct, and any later error is the ML model's fault, not a bug."]]));
kids.push(P("First real result (10 unseen waves, real viscosity). \"Tail error\" = how wrong, averaged over the future part (0 = perfect). \"Benefit\" = how much the hybrid improves on pure FNO."));
kids.push(tbl([1360,2200,1900,1700,1300,900], [
  ["Switch t_s","FNO wave error at handover","Pure-FNO future error","Hybrid future error","Benefit","Waves improved"],
  ["1.0","1.4%","14.1%","1.1%","92%","10/10"],
  ["1.2","3.2%","16.9%","2.1%","86%","10/10"],
  ["1.4","7.7%","20.9%","7.5%","62%","10/10"],
  ["1.6","16%","25.4%","16.3%","34%","10/10"],
  ["1.8","25%","30.1%","25.0%","16%","9/10"],
]));
kids.push(P("What this table means (three conclusions):"));
kids.push(bullet("The hybrid works, strongly, for early switches: at t=1.0 it cuts error from 14.1% to 1.1% on all 10 waves."));
kids.push(bullet("Earlier is better - a viability window: benefit falls as we wait (92% -> 16%), because the handed-over wave is already more wrong."));
kids.push(bullet("Deepest point: the hybrid's error is almost exactly the FNO's error at handover, and the perfect-handoff yardstick is near zero. So the numerical solver adds almost no error - it stops error growing but cannot undo error already there."));

// 6c
kids.push(H1("6c. Verification - my restart tool is exactly the team's solver"));
kids.push(P("A fair worry: \"your restart solver is your own copy - is it really the same as the team's official solver?\" We answered it by running the team's actual solver code side-by-side with our restart tool."));
kids.push(bullet("They are identical - the difference was exactly zero (bit-for-bit) on every test wave. Our tool computes the same numbers, just with the extra ability to start from the middle."));
kids.push(bullet("We can regenerate the test waves - our 10 held-out waves are exactly the project's test waves (900-909), so we can make more whenever we want."));
kids.push(quote("Say in the viva: I proved my restart solver is bit-for-bit identical to the team's official solver, so the hybrid's numerical step is fully trusted."));

// 7 novelty
kids.push(new Paragraph({ pageBreakBefore:true, heading:HeadingLevel.HEADING_1, children:[new TextRun("7. The novelty - explained through the code (important for the viva)")] }));
kids.push(PR([["What is NOT novel: ",{bold:true}],["connecting an ML model to a numerical solver is a known idea (papers such as ANCHOR and PDE-Refiner exist). We do not claim to invent hybrid solving."]]));
kids.push(P("What IS novel here, and where it lives in the code:"));
kids.push(PR([["1) A working, solver-agnostic restart bridge - ",{bold:true}],["restart_spectral.py, function solve_from. The team's solver could only start from the beginning. My solve_from(u0, i_start) lets it continue from any ML-produced wave at any moment. That single capability is what makes a hybrid possible at all, and it accepts any wave, so the same bridge works for FNO, DeepONet or PINN."]]));
kids.push(PR([["2) A fair, self-checking measurement design - ",{bold:true}],["make_figures.py. The novelty is not just \"it works\" but that we can prove when and why. The true-state yardstick separates \"error the ML brought in\" from \"error the solver added\"; the switch-time sweep over many waves turns the viability window into a measured curve, not a lucky single example."]]));
kids.push(quote("One-line novelty: My novelty is a validated, solver-agnostic handoff plus a measurement that quantifies exactly when it helps. In code: solve_from makes the handoff possible, and the true-state yardstick proves the numerical part adds no error - so the remaining error is purely inherited from the ML model. That turns \"it seems to work\" into \"here is the boundary where it works, with proof.\""));

// 8 figures
kids.push(new Paragraph({ pageBreakBefore:true, heading:HeadingLevel.HEADING_1, children:[new TextRun("8. Figures - what each one shows")] }));

kids.push(H3("Figure 1 - Error over time"));
kids.push(figure("fig1_error_over_time.png"));
kids.push(P("Three lines over time, averaged across the 10 unseen waves (shaded = spread). FNO alone (red) is low until t=1 then shoots up to ~35%; Numerical alone (blue) stays near zero; Hybrid (green) follows the FNO early, then after the handoff stays flat and low. Conclusion: the handoff stops the model's error blow-up."));
kids.push(quote("Say: Red is the fast model failing in the future; green is my hybrid staying accurate by handing over to blue."));

kids.push(H3("Figure 2 - When to switch vs how much it helps"));
kids.push(figure("fig2_switch_time_vs_benefit.png"));
kids.push(P("x = handoff time; y = average benefit (error removed vs pure FNO), with error bars. Benefit is highest for an early handoff (~92% at t=1.0) and falls to ~16% at t=1.8; the dashed line is our pre-set 10% \"worth it\" bar. Conclusion: there is a viability window - switch early, win big."));
kids.push(quote("Say: This curve is my main finding - earlier handoff helps more."));

kids.push(H3("Figure 3 - Why late handoffs help less"));
kids.push(figure("fig3_handoff_error_vs_benefit.png"));
kids.push(P("One dot per wave per handoff time. x = how wrong the FNO wave already is at handoff; y = benefit. Dots drift down-right. Conclusion: benefit is controlled by the quality of the wave at handoff."));
kids.push(quote("Say: Benefit depends on how good the ML state is when I hand it over."));

kids.push(H3("Figure 4 - Proof the numerical part is clean"));
kids.push(figure("fig4_hybrid_vs_upper_bound.png"));
kids.push(P("Log scale. Pure FNO (red) highest; Hybrid from the FNO state (green) lower; Upper bound - restart from the TRUE state (grey) sits near 0.000001; the black dashed line is the FNO's error at handoff. The green line lies almost exactly on the black dashed line. Conclusion (key point): numerical continuation adds almost no error of its own, so the hybrid's error is the error it inherited from the FNO state - the solver stops growth but cannot undo existing error."));
kids.push(quote("Say: My true-state yardstick proves the numerical step is clean; the remaining error came from the ML model."));

kids.push(H3("Figure 5 - Accuracy vs cost"));
kids.push(figure("fig5_accuracy_vs_cost.png"));
kids.push(P("x = fraction of the trajectory done by the numerical solver (a cost stand-in); y = hybrid error. Earlier handoff = more numerical work = lower error. Conclusion: a clean accuracy-vs-cost trade-off - exactly the input a cost controller (teammate's module) needs."));
kids.push(quote("Say: Earlier switching costs more compute but buys accuracy; that trade-off is what the controller will optimise."));

// 9 code
kids.push(new Paragraph({ pageBreakBefore:true, heading:HeadingLevel.HEADING_1, children:[new TextRun("9. The code, explained simply")] }));
kids.push(PR([["restart_spectral.py - the \"continuer\". ",{bold:true}],["Re-creates the team's exact numerical recipe but lets it start from any wave at any time and run to the end. It integrates internally with tiny steps for stability but records the answer on the same time points as the dataset, so all methods compare fairly. Key function: solve_from(u0, i_start)."]]));
kids.push(PR([["make_figures.py - the experiment and figures. ",{bold:true}],["For each wave and each switch time it builds three trajectories - pure FNO, hybrid (restart from the FNO wave), and upper bound (restart from the true wave) - measures how wrong each is versus the Cole-Hopf truth, writes a summary table, and renders the five figures."]]));
kids.push(PR([["verify_restart.py - the proof. ",{bold:true}],["Runs the team's real solver code next to our restart tool and shows they are identical, and regenerates the test waves to confirm which waves we used."]]));

// 10 next
kids.push(H1("10. Where we are and what's next"));
kids.push(P("Done: the audit and data contract; the restart tool (proven identical to the team solver); the safety gate; the first hybrid result on 10 waves; the five figures."));
kids.push(P("Next (phase plan): extend from 10 to 20 waves and add DeepONet; add a second \"shape\" error metric alongside relative error; a small filtering study; automated tests; and the written thesis section. The full phase plan with time allocation is in PROJECT_PLAN.md."));

// 11 change log
kids.push(H1("11. Change log"));
kids.push(tbl([1700,7660], [
  ["Date","Update"],
  ["2026-07-08","Initial guide: audit (direct-map FNO), restart gate, first 10-wave hybrid result, novelty."],
  ["2026-07-08","Figures generated and explained; professional naming; phase-based plan added."],
  ["2026-07-08","Phase 1 verification: restart tool proven bit-for-bit identical to team solver; Word version of this guide created."],
]));

const doc = new Document({
  styles: {
    default: { document: { run: { font:"Arial", size:22 } } },
    paragraphStyles: [
      { id:"Title", name:"Title", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:40, bold:true, font:"Arial", color:"1F3864" }, paragraph:{ spacing:{after:240} } },
      { id:"Heading1", name:"Heading 1", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:30, bold:true, font:"Arial", color:"1F3864" }, paragraph:{ spacing:{before:260,after:140}, outlineLevel:0 } },
      { id:"Heading2", name:"Heading 2", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:26, bold:true, font:"Arial", color:"2E75B6" }, paragraph:{ spacing:{before:200,after:120}, outlineLevel:1 } },
      { id:"Heading3", name:"Heading 3", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:23, bold:true, font:"Arial", color:"2E75B6" }, paragraph:{ spacing:{before:160,after:80}, outlineLevel:2 } },
    ]
  },
  numbering:{ config:[{ reference:"bullets", levels:[{ level:0, format:"bullet", text:"•", alignment:AlignmentType.LEFT, style:{ paragraph:{ indent:{ left:540, hanging:280 } } } }] }] },
  sections: [{
    properties:{ page:{ size:{ width:12240, height:15840 }, margin:{ top:1440,right:1440,bottom:1440,left:1440 } } },
    children: kids
  }]
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, b.length, "bytes"); });
