import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import {
  getMe,
  googleLogin as apiGoogleLogin,
  requestOtp as apiRequestOtp,
  verifyOtp as apiVerifyOtp,
} from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Restore session from localStorage on mount
  useEffect(() => {
    const token = localStorage.getItem('rag_token');
    if (!token) { setLoading(false); return; }
    getMe().then((result) => {
      if (result.data) setUser(result.data);
      else localStorage.removeItem('rag_token');
      setLoading(false);
    });
  }, []);

  const requestOtp = useCallback(async (email) => apiRequestOtp(email), []);

  const verifyOtp = useCallback(async (email, otp) => {
    const result = await apiVerifyOtp(email, otp);
    if (result.error) return result;
    localStorage.setItem('rag_token', result.data.access_token);
    setUser(result.data.user);
    return { data: result.data.user };
  }, []);

  const loginWithGoogle = useCallback(async (credential) => {
    const result = await apiGoogleLogin(credential);
    if (result.error) return result;
    localStorage.setItem('rag_token', result.data.access_token);
    setUser(result.data.user);
    return { data: result.data.user };
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('rag_token');
    setUser(null);
  }, []);

  const refreshUser = useCallback(async () => {
    const result = await getMe();
    if (result.data) setUser(result.data);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, requestOtp, verifyOtp, loginWithGoogle, logout, refreshUser }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
}
