import { useState } from 'react';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { AuthProvider, useAuth } from './context/AuthContext';
import LoginPage from './pages/LoginPage';
import ProfilePage from './pages/ProfilePage';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import QueryInput from './components/QueryInput';
import { useCollections } from './hooks/useCollections';
import { useChat } from './hooks/useChat';

function AppShell() {
  const { user, logout } = useAuth();
  const { collections, refresh, remove } = useCollections();
  const { messages, loading, sendQuery, clearChat } = useChat();
  const [selectedCollection, setSelectedCollection] = useState('default');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showProfile, setShowProfile] = useState(false);

  function handleCollectionSelect(name) {
    setSelectedCollection(name);
    setSidebarOpen(false);
  }

  return (
    <div className="flex h-screen bg-[#0a0a0f] overflow-hidden">
      <Sidebar
        collections={collections}
        selectedCollection={selectedCollection}
        onCollectionSelect={handleCollectionSelect}
        onRefresh={refresh}
        onDelete={remove}
        sidebarOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(v => !v)}
        user={user}
        onLogout={logout}
        onOpenProfile={() => setShowProfile(true)}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <header className="flex items-center gap-3 px-4 py-3 border-b border-slate-800 bg-slate-900/60 backdrop-blur shrink-0">
          <button
            onClick={() => setSidebarOpen(v => !v)}
            className="lg:hidden text-slate-400 hover:text-slate-200"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <span className="text-sm font-medium text-slate-300 truncate">{selectedCollection}</span>
            <span className="text-slate-600 text-xs hidden sm:block">· Multimodal RAG</span>
          </div>
          {/* Profile button in header (mobile-friendly) */}
          <button
            onClick={() => setShowProfile(true)}
            className="w-7 h-7 rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-xs font-bold shrink-0"
            title="Profile & Settings"
          >
            {(user?.display_name || user?.email || 'U')[0].toUpperCase()}
          </button>
        </header>

        <ChatWindow messages={messages} loading={loading} onClear={clearChat} />

        <QueryInput
          onSubmit={sendQuery}
          loading={loading}
          collections={collections}
          selectedCollection={selectedCollection}
          onCollectionChange={setSelectedCollection}
        />
      </div>

      {showProfile && <ProfilePage onClose={() => setShowProfile(false)} />}
    </div>
  );
}

function AuthGate() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-violet-500/30 border-t-violet-500 rounded-full animate-spin" />
      </div>
    );
  }

  if (!user) return <LoginPage />;
  return <AppShell />;
}

export default function App() {
  const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';
  return (
    <GoogleOAuthProvider clientId={clientId}>
      <AuthProvider>
        <AuthGate />
      </AuthProvider>
    </GoogleOAuthProvider>
  );
}
