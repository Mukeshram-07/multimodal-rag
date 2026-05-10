import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { updateMe } from '../services/api';
import { useCollections } from '../hooks/useCollections';

function StatCard({ label, value }) {
  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4 text-center">
      <div className="text-2xl font-bold text-violet-400">{value}</div>
      <div className="text-xs text-slate-500 mt-1">{label}</div>
    </div>
  );
}

function ProviderBadge({ provider }) {
  if (provider === 'google') {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs bg-blue-500/15 text-blue-300 border border-blue-500/30 rounded-full px-2.5 py-0.5">
        <svg className="w-3 h-3" viewBox="0 0 24 24" fill="currentColor">
          <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
          <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
          <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
          <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
        </svg>
        Google OAuth
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-xs bg-violet-500/15 text-violet-300 border border-violet-500/30 rounded-full px-2.5 py-0.5">
      ✉ Email OTP
    </span>
  );
}

export default function ProfilePage({ onClose }) {
  const { user, logout, refreshUser } = useAuth();
  const { collections } = useCollections();
  const [displayName, setDisplayName] = useState(user?.display_name || '');
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState('');

  const initials = (user?.display_name || user?.email || 'U')[0].toUpperCase();
  const joinDate = user?.created_at
    ? new Date(user.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
    : 'Unknown';

  async function handleSave(e) {
    e.preventDefault();
    setSaving(true);
    setSaveMsg('');
    const result = await updateMe(displayName);
    setSaving(false);
    if (result.error) {
      setSaveMsg('Failed to save: ' + result.error.message);
    } else {
      await refreshUser();
      setSaveMsg('Saved!');
      setTimeout(() => setSaveMsg(''), 2000);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-lg shadow-2xl overflow-y-auto max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <h2 className="text-base font-semibold text-slate-200">Profile & Settings</h2>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 text-xl leading-none">×</button>
        </div>

        <div className="p-6 space-y-6">
          {/* Avatar + identity */}
          <div className="flex items-center gap-4">
            {user?.avatar_url ? (
              <img
                src={user.avatar_url}
                alt={user.display_name}
                className="w-16 h-16 rounded-2xl object-cover shadow-lg shrink-0"
              />
            ) : (
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-2xl font-bold shadow-lg shrink-0">
                {initials}
              </div>
            )}
            <div>
              <div className="text-lg font-semibold text-slate-100">{user?.display_name || 'User'}</div>
              <div className="text-sm text-slate-400">{user?.email}</div>
              <div className="flex items-center gap-2 mt-1.5">
                <ProviderBadge provider={user?.auth_provider || 'otp'} />
              </div>
              <div className="text-xs text-slate-600 mt-1">Member since {joinDate}</div>
            </div>
          </div>

          {/* Usage stats */}
          <div>
            <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">Usage</div>
            <div className="grid grid-cols-3 gap-3">
              <StatCard label="Collections" value={collections.length} />
              <StatCard label="Queries" value="—" />
              <StatCard label="PDFs" value="—" />
            </div>
          </div>

          {/* Settings */}
          <div>
            <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">Settings</div>
            <form onSubmit={handleSave} className="space-y-3">
              <div>
                <label className="block text-xs text-slate-400 mb-1.5">Display name</label>
                <input
                  type="text"
                  value={displayName}
                  onChange={e => setDisplayName(e.target.value)}
                  className="w-full bg-slate-800/60 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-violet-500"
                />
              </div>
              <div className="flex items-center gap-3">
                <button
                  type="submit"
                  disabled={saving}
                  className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white text-sm font-medium"
                >
                  {saving ? 'Saving…' : 'Save changes'}
                </button>
                {saveMsg && (
                  <span className={`text-sm ${saveMsg.startsWith('Failed') ? 'text-red-400' : 'text-emerald-400'}`}>
                    {saveMsg}
                  </span>
                )}
              </div>
            </form>
          </div>

          {/* Security */}
          <div>
            <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">Security</div>
            <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 space-y-2 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Authentication</span>
                <ProviderBadge provider={user?.auth_provider || 'otp'} />
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Session</span>
                <span className="text-slate-300">JWT · 24h expiry</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Account ID</span>
                <span className="text-slate-500 font-mono text-xs">{user?.id?.slice(0, 8)}…</span>
              </div>
            </div>
          </div>

          {/* Danger zone */}
          <div className="border-t border-slate-800 pt-4">
            <button
              onClick={logout}
              className="w-full py-2.5 rounded-xl border border-red-500/30 text-red-400 hover:bg-red-500/10 text-sm font-medium transition-colors"
            >
              Sign out
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
