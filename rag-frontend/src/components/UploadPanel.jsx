import { useState, useRef } from 'react';
import { ingestPDF } from '../services/api';

export default function UploadPanel({ collections, onIngested }) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [collection, setCollection] = useState('default');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null); // { success, message }
  const inputRef = useRef();

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f?.type === 'application/pdf') setFile(f);
  }

  async function handleIngest() {
    if (!file) return;
    setLoading(true);
    setResult(null);
    const bytes = await file.arrayBuffer();
    const res = await ingestPDF(bytes, file.name, collection || 'default');
    setLoading(false);
    if (res.error) {
      setResult({ success: false, message: res.error.message });
    } else {
      setResult({ success: true, message: `✓ ${res.data.chunk_count} chunks stored in "${res.data.collection_name}"` });
      setFile(null);
      onIngested?.();
    }
  }

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          relative border-2 border-dashed rounded-xl p-4 text-center cursor-pointer
          transition-colors duration-200
          ${dragging ? 'border-violet-400 bg-violet-500/10' : 'border-slate-600 hover:border-slate-500 bg-slate-800/40'}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={e => setFile(e.target.files[0])}
        />
        {file ? (
          <div className="flex items-center gap-2 justify-center text-sm text-slate-300">
            <span className="text-lg">📄</span>
            <span className="truncate max-w-[140px]">{file.name}</span>
            <button
              onClick={e => { e.stopPropagation(); setFile(null); }}
              className="text-slate-500 hover:text-red-400 ml-1"
            >✕</button>
          </div>
        ) : (
          <div className="text-slate-500 text-xs">
            <div className="text-2xl mb-1">📂</div>
            Drop PDF or click
          </div>
        )}
      </div>

      {/* Collection input */}
      <input
        type="text"
        value={collection}
        onChange={e => setCollection(e.target.value)}
        placeholder="Collection name"
        className="w-full bg-slate-800/60 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-violet-500"
      />

      {/* Ingest button */}
      <button
        onClick={handleIngest}
        disabled={!file || loading}
        className="w-full py-2 rounded-lg text-sm font-medium bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed text-white"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Ingesting…
          </span>
        ) : 'Ingest PDF'}
      </button>

      {/* Result toast */}
      {result && (
        <div className={`text-xs rounded-lg px-3 py-2 ${result.success ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30' : 'bg-red-500/15 text-red-300 border border-red-500/30'}`}>
          {result.message}
        </div>
      )}
    </div>
  );
}
