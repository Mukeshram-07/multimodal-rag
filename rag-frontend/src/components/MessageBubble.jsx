import ReactMarkdown from 'react-markdown';
import CitationPanel from './CitationPanel';

function UserBubble({ content }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[75%] bg-violet-600/80 text-white rounded-2xl rounded-tr-sm px-4 py-3 text-sm leading-relaxed shadow-lg">
        {content}
      </div>
    </div>
  );
}

function AssistantBubble({ content, citations }) {
  return (
    <div className="flex gap-3">
      <div className="w-7 h-7 rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-xs font-bold shrink-0 mt-0.5 shadow">
        R
      </div>
      <div className="flex-1 min-w-0">
        <div className="bg-slate-800/70 border border-slate-700/50 rounded-2xl rounded-tl-sm px-4 py-3 text-sm leading-relaxed shadow">
          <div className="prose prose-invert prose-sm max-w-none
            prose-p:my-1 prose-headings:text-slate-200 prose-code:text-violet-300
            prose-code:bg-slate-900 prose-code:px-1 prose-code:rounded
            prose-pre:bg-slate-900 prose-pre:border prose-pre:border-slate-700">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        </div>
        <CitationPanel citations={citations} />
      </div>
    </div>
  );
}

function ErrorBubble({ content }) {
  return (
    <div className="flex gap-3">
      <div className="w-7 h-7 rounded-full bg-red-500/20 border border-red-500/40 flex items-center justify-center text-xs shrink-0 mt-0.5">
        !
      </div>
      <div className="bg-red-500/10 border border-red-500/30 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-red-300">
        {content}
      </div>
    </div>
  );
}

export default function MessageBubble({ message }) {
  if (message.role === 'user') return <UserBubble content={message.content} />;
  if (message.role === 'error') return <ErrorBubble content={message.content} />;
  return <AssistantBubble content={message.content} citations={message.citations} />;
}
