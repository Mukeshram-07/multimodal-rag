import { useState, useEffect } from 'react';
import { checkHealth } from '../services/api';

export default function StatusBadge() {
  const [status, setStatus] = useState('checking'); // 'online' | 'offline' | 'checking'

  useEffect(() => {
    let cancelled = false;
    async function ping() {
      const result = await checkHealth();
      if (!cancelled) setStatus(result.error ? 'offline' : 'online');
    }
    ping();
    const id = setInterval(ping, 30000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const dot = status === 'online'
    ? 'bg-emerald-400'
    : status === 'offline'
    ? 'bg-red-400'
    : 'bg-yellow-400 animate-pulse';

  const label = status === 'online' ? 'API online' : status === 'offline' ? 'API offline' : 'Connecting…';

  return (
    <div className="flex items-center gap-1.5 text-xs text-slate-400">
      <span className={`w-2 h-2 rounded-full ${dot}`} />
      {label}
    </div>
  );
}
