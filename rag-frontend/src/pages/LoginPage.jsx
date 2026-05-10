import { useEffect, useRef, useState } from 'react';
import { GoogleLogin } from '@react-oauth/google';
import OtpInput from '../components/OtpInput';
import { useAuth } from '../context/AuthContext';

const RESEND_COOLDOWN = 60;

export default function LoginPage() {
  const { requestOtp, verifyOtp, loginWithGoogle } = useAuth();

  const [step, setStep] = useState('email'); // 'email' | 'otp'
  const [email, setEmail] = useState('');
  const [otp, setOtp] = useState('');
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [countdown, setCountdown] = useState(0);
  const timerRef = useRef(null);

  function startCountdown() {
    setCountdown(RESEND_COOLDOWN);
    timerRef.current = setInterval(() => {
      setCountdown(c => {
        if (c <= 1) {
          clearInterval(timerRef.current);
          return 0;
        }
        return c - 1;
      });
    }, 1000);
  }

  useEffect(() => () => clearInterval(timerRef.current), []);

  async function handleRequestOtp(e) {
    e?.preventDefault();

    setError('');
    setLoading(true);

    const result = await requestOtp(email);

    setLoading(false);

    if (result.error) {
      setError(result.error.message);
    } else {
      setStep('otp');
      setSuccess(`Code sent to ${email}`);
      startCountdown();
    }
  }

  async function handleVerifyOtp(e) {
    e?.preventDefault();

    if (otp.length !== 6) return;

    setError('');
    setLoading(true);

    const result = await verifyOtp(email, otp);

    setLoading(false);

    if (result.error) {
      setError(result.error.message);
      setOtp('');
    }
  }

  async function handleGoogleSuccess(credentialResponse) {
    console.log("Google success response:", credentialResponse);

    setError('');
    setGoogleLoading(true);

    const credential = credentialResponse?.credential;

    console.log("Credential:", credential);

    if (!credential) {
      console.error("No credential received from Google");
      setError("Google did not return a credential.");
      setGoogleLoading(false);
      return;
    }

    console.log("Sending credential to backend...");

    const result = await loginWithGoogle(credential);

    console.log("Backend login result:", result);

    setGoogleLoading(false);

    if (result.error) {
      console.error(result.error);
      setError(result.error.message);
    }
  }

  async function handleResend() {
    if (countdown > 0) return;

    setError('');
    setOtp('');

    await handleRequestOtp();
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-2xl font-bold shadow-xl mb-3">
            R
          </div>

          <h1 className="text-2xl font-semibold text-slate-100">
            {step === 'email'
              ? 'Sign in to RAG'
              : 'Check your email'}
          </h1>

          <p className="text-slate-500 text-sm mt-1">
            {step === 'email'
              ? 'Enter your email or continue with Google'
              : `We sent a 6-digit code to ${email}`}
          </p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-8 shadow-2xl backdrop-blur">
          {step === 'email' ? (
            <div className="space-y-5">

              {/* Google Sign-In */}
              <div className="flex flex-col items-center gap-3">
                {googleLoading ? (
                  <div className="flex items-center gap-2 text-sm text-slate-400">
                    <span className="w-4 h-4 border-2 border-violet-500/30 border-t-violet-500 rounded-full animate-spin" />
                    Signing in with Google…
                  </div>
                ) : (
                  <GoogleLogin
                    onSuccess={handleGoogleSuccess}
                    onError={() => {
                      console.error("Google sign-in failed");
                      setError('Google sign-in failed. Please try again.');
                    }}
                    theme="filled_black"
                    shape="rectangular"
                    size="large"
                    text="continue_with"
                    width="100%"
                  />
                )}
              </div>

              {/* Divider */}
              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-slate-800" />
                <span className="text-xs text-slate-600">
                  or continue with email
                </span>
                <div className="flex-1 h-px bg-slate-800" />
              </div>

              {/* Email OTP Form */}
              <form
                onSubmit={handleRequestOtp}
                className="space-y-4"
              >
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1.5">
                    Email address
                  </label>

                  <input
                    type="email"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    required
                    autoComplete="email"
                    placeholder="you@example.com"
                    className="w-full bg-slate-800/60 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500/30"
                  />
                </div>

                {error && (
                  <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2.5 text-sm text-red-300">
                    {error}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={loading || !email}
                  className="w-full py-3 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium text-sm transition-colors"
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Sending code…
                    </span>
                  ) : (
                    'Send verification code →'
                  )}
                </button>
              </form>
            </div>
          ) : (
            <form
              onSubmit={handleVerifyOtp}
              className="space-y-6"
            >
              {success && (
                <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-4 py-2.5 text-sm text-emerald-300 text-center">
                  {success}
                </div>
              )}

              <OtpInput
                value={otp}
                onChange={setOtp}
                disabled={loading}
              />

              {error && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2.5 text-sm text-red-300 text-center">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading || otp.length !== 6}
                className="w-full py-3 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium text-sm transition-colors"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Verifying…
                  </span>
                ) : (
                  'Verify code'
                )}
              </button>

              <div className="flex items-center justify-between text-sm">
                <button
                  type="button"
                  onClick={() => {
                    setStep('email');
                    setError('');
                    setOtp('');
                  }}
                  className="text-slate-500 hover:text-slate-300"
                >
                  ← Change email
                </button>

                <button
                  type="button"
                  onClick={handleResend}
                  disabled={countdown > 0}
                  className="text-violet-400 hover:text-violet-300 disabled:text-slate-600 disabled:cursor-not-allowed"
                >
                  {countdown > 0
                    ? `Resend in ${countdown}s`
                    : 'Resend code'}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
