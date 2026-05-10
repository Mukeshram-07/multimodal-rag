import { useState } from 'react';
import StatusBadge from './StatusBadge';
import UploadPanel from './UploadPanel';

export default function Sidebar({ collections, onCollectionSelect, selectedCollection, onRefresh, onDelete, sidebarOpen, onToggle, user, onLogout, onOpenProfile }) {
  const [confirmDelete, setConfirmDelete] = useState(null);

  async function handleDelete(name) {
    if (confirmDelete === name) {
      await onDelete(name);
      setConfirmDelete(null);
      if (selectedCollection === name) onCollectionSelect(collections.find(c => c !== name) || 'default');
    } else {
      setConfirmDelete(name);
      setTimeout(() => setConfirmDelete(null), 3000);
    }
  }

  return (
    <>
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-20 lg:hidden"
          onClick={onToggle}
        />
      )}

      <aside className={`
        fixed lg:relative inset-y-0 left-0 z-30 lg:z-auto
        w-72 flex flex-col bg-slate-900/95 border-r border-slate-800
        transform transition-transform duration-300 ease-in-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-sm font-bold shadow">
              R
            </div>
            <span className="font-semibold text-slate-200 text-sm">RAG System</span>
          </div>
          <button onClick={onToggle} className="lg:hidden text-slate-500 hover:text-slate-300">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
          {/* Upload */}
          <section>
            <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2 px-1">Upload PDF</div>
            <UploadPanel collections={collections} onIngested={onRefresh} />
          </section>

          {/* Collections */}
          <section>
            <div className="flex items-center justify-between mb-2 px-1">
              <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">Collections</span>
              <button onClick={onRefresh} className="text-slate-600 hover:text-slate-400" title="Refresh">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            </div>

            {collections.length === 0 ? (
              <p className="text-xs text-slate-600 px-1">No collections yet.</p>
            ) : (
              <div className="space-y-0.5">
                {collections.map(name => (
                  <div
                    key={name}
                    className={`
                      group flex items-center justify-between rounded-lg px-3 py-2 cursor-pointer
                      ${selectedCollection === name
                        ? 'bg-violet-600/20 border border-violet-500/30 text-violet-300'
                        : 'hover:bg-slate-800/60 text-slate-400 hover:text-slate-200 border border-transparent'}
                    `}
                    onClick={() => onCollectionSelect(name)}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <svg className="w-3.5 h-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                      <span className="text-xs truncate">{name}</span>
                    </div>
                    <button
                      onClick={e => { e.stopPropagation(); handleDelete(name); }}
                      className={`shrink-0 text-xs rounded px-1.5 py-0.5 opacity-0 group-hover:opacity-100
                        ${confirmDelete === name
                          ? 'bg-red-500/20 text-red-400 border border-red-500/40 opacity-100'
                          : 'text-slate-600 hover:text-red-400'}`}
                      title={confirmDelete === name ? 'Click again to confirm' : 'Delete'}
                    >
                      {confirmDelete === name ? 'Sure?' : '✕'}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-slate-800 space-y-2">
          {user && (
            <button
              onClick={onOpenProfile}
              className="w-full flex items-center gap-2 rounded-lg px-2 py-2 hover:bg-slate-800/60 group"
            >
              <div className="w-7 h-7 rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-xs font-bold shrink-0">
                {(user.display_name || user.email)[0].toUpperCase()}
              </div>
              <div className="flex-1 min-w-0 text-left">
                <div className="text-xs font-medium text-slate-300 truncate">{user.display_name || user.email}</div>
                <div className="text-xs text-slate-600 truncate">{user.email}</div>
              </div>
              <svg className="w-3.5 h-3.5 text-slate-600 group-hover:text-slate-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>
          )}
          <StatusBadge />
        </div>
      </aside>
    </>
  );
}
