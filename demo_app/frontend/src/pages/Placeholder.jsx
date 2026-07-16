import { Card } from "../components/ui.jsx";

export default function Placeholder({ title, group, note }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-slate-400 dark:text-slate-500">{group}</div>
      <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mt-1">{title}</h1>
      <Card className="mt-4">
        <p className="text-sm text-slate-600 dark:text-slate-300">{note}</p>
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-3">
          This section is a placeholder in the starter. Build it as its own page component in
          <code className="bg-slate-100 dark:bg-slate-700 px-1 rounded mx-1">src/pages/</code> and it is already wired into the
          sidebar. Fetch data from a matching backend route so every section stays independent.
        </p>
      </Card>
    </div>
  );
}
