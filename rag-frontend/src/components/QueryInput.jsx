import { useState, useRef } from 'react';

const EXAMPLE_PROMPTS = [
  'What is the main topic of this document?',
  'Summarize the key findings.',
  'What conclusions does the author draw?',
  'List the main recommendations.',
];

export default function QueryInput({ onSubmit, loading, collections, selectedCollection, onCollectionChange }) {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [filterSource, setFilterSource] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const textareaRef = useRef();

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    const q = query.trim();
    if (!q || loading) return;
    onSubmit(q, selectedCollection, topK, filterSource.trim() || null);
    setQuery('');
  }

  return (
    <div className="border-t border-slate-800 bg-slate-900/80 backdrop-blur px-4 py-3">
      {/* Advanced options */}
      <div className="flex items-center gap-3 mb-2 flex-wrap">
        <select
          value={selectedCollection}
          onChange={e => onCollectionChange(e.target.value)}
          className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-xs text-slate-300 focus:outline-none focus:border-violet-500"
        >
          {collections.length === 0 && <option value="default">default</option>}
          {collections.map(c => <option key={c} value={c}>{c}</option>)}
        </select>

        <button
          onClick={() => setShowAdvanced(v => !v)}
          className="text-xs text-slate-500 hover:text-slate-300"
        >
          {showAdvanced ? '▲ Less' : '▼ Options'}
        </button>

        {showAdvanced && (
          <>
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-slate-500">Top-K</span>
              <input
                type="number"
                min={1} max={20}
                value={topK}
                onChange={e => setTopK(Number(e.target.value))}
                className="w-14 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-300 focus:outline-none focus:border-violet-500"
              />
            </div>
            <input
              type="text"
              value={filterSource}
              onChange={e => setFilterSource(e.target.value)}
              placeholder="Filter source…"
              className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-xs text-slate-300 placeholder-slate-600 focus:outline-none focus:border-violet-500 w-36"
            />
          </>
        )}
      </div>

      {/* Input row */}
      <div className="flex gap-2 items-end">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
            rows={1}
            disabled={loading}
            className="w-full bg-slate-800/80 border border-slate-700 rounded-xl px-4 py-3 pr-12 text-sm text-slate-200 placeholder-slate-500 resize-none focus:outline-none focus:border-violet-500 disabled:opacity-50 leading-relaxed"
            style={{ minHeight: '48px', maxHeight: '160px', overflowY: 'auto' }}
            onInput={e => {
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
            }}
          />
        </div>
        <button
          onClick={submit}
          disabled={!query.trim() || loading}
          className="h-12 w-12 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center shrink-0 shadow-lg"
        >
          {loading ? (
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          )}
        </button>
      </div>

      {/* Example prompts (shown when empty) */}
      {!query && (
        <div className="flex gap-2 mt-2 flex-wrap">
          {EXAMPLE_PROMPTS.map(p => (
            <button
              key={p}
              onClick={() => setQuery(p)}
              className="text-xs text-slate-500 hover:text-slate-300 bg-slate-800/60 hover:bg-slate-800 border border-slate-700/50 rounded-full px-3 py-1"
            >
              {p}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
