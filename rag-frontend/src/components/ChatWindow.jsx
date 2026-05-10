import { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';

function WelcomeScreen() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-8 select-none">
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-3xl mb-4 shadow-xl">
        🔍
      </div>
      <h2 className="text-xl font-semibold text-slate-200 mb-2">Ask your documents</h2>
      <p className="text-slate-500 text-sm max-w-sm leading-relaxed">
        Upload a PDF in the sidebar, then ask questions below. Answers are grounded in your documents with citations.
      </p>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="w-7 h-7 rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-xs font-bold shrink-0 shadow">
        R
      </div>
      <div className="bg-slate-800/70 border border-slate-700/50 rounded-2xl rounded-tl-sm px-4 py-3">
        <div className="flex gap-1 items-center h-4">
          {[0, 1, 2].map(i => (
            <span
              key={i}
              className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"
              style={{ animationDelay: `${i * 150}ms` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export default function ChatWindow({ messages, loading, onClear }) {
  const bottomRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Toolbar */}
      {messages.length > 0 && (
        <div className="flex justify-end px-4 py-2 border-b border-slate-800/60">
          <button
            onClick={onClear}
            className="text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            Clear chat
          </button>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && !loading ? (
          <WelcomeScreen />
        ) : (
          <>
            {messages.map(msg => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {loading && <TypingIndicator />}
          </>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
