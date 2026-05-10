import { useState } from 'react';

export default function CitationPanel({ citations }) {
  const [expanded, setExpanded] = useState(false);

  if (!citations?.length) return null;

  return (
    <div className="mt-3">
      <button
        onClick={() => setExpanded(v => !v)}
        className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 mb-2"
      >
        <span className={`transition-transform ${expanded ? 'rotate-90' : ''}`}>▶</span>
        {citations.length} source{citations.length !== 1 ? 's' : ''}
      </button>

      {expanded && (
        <div className="space-y-1.5">
          {citations.map((c, i) => (
            <div
              key={i}
              className="flex items-center gap-3 bg-slate-800/60 border border-slate-700/60 rounded-lg px-3 py-2 text-xs hover:border-violet-500/40 hover:bg-slate-800"
            >
              <span className="text-violet-400 font-mono font-bold w-5 text-center">{i + 1}</span>
              <div className="flex-1 min-w-0">
                <div className="text-slate-200 truncate font-medium">{c.source}</div>
                <div className="text-slate-500">p.{c.page} · chunk {c.chunk_index}</div>
              </div>
              <div className="text-right shrink-0">
                <div className="text-emerald-400 font-mono">{(c.score * 100).toFixed(1)}%</div>
                <div className="text-slate-600">score</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
