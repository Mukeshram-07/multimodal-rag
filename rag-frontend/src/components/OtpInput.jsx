import { useEffect, useRef } from 'react';

/**
 * 6-digit OTP input with auto-focus, auto-advance, and paste support.
 */
export default function OtpInput({ value, onChange, disabled }) {
  const digits = (value + '      ').slice(0, 6).split('');
  const refs = Array.from({ length: 6 }, () => useRef(null));

  useEffect(() => {
    refs[0]?.current?.focus();
  }, []);

  function handleChange(i, e) {
    const char = e.target.value.replace(/\D/g, '').slice(-1);
    const next = value.slice(0, i) + char + value.slice(i + 1);
    onChange(next.replace(/\s/g, ''));
    if (char && i < 5) refs[i + 1]?.current?.focus();
  }

  function handleKeyDown(i, e) {
    if (e.key === 'Backspace') {
      if (!value[i] && i > 0) {
        refs[i - 1]?.current?.focus();
        onChange(value.slice(0, i - 1) + value.slice(i));
      } else {
        onChange(value.slice(0, i) + value.slice(i + 1));
      }
    } else if (e.key === 'ArrowLeft' && i > 0) {
      refs[i - 1]?.current?.focus();
    } else if (e.key === 'ArrowRight' && i < 5) {
      refs[i + 1]?.current?.focus();
    }
  }

  function handlePaste(e) {
    e.preventDefault();
    const pasted = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6);
    onChange(pasted);
    const focusIdx = Math.min(pasted.length, 5);
    refs[focusIdx]?.current?.focus();
  }

  return (
    <div className="flex gap-2 justify-center">
      {digits.map((d, i) => (
        <input
          key={i}
          ref={refs[i]}
          type="text"
          inputMode="numeric"
          maxLength={1}
          value={value[i] || ''}
          onChange={(e) => handleChange(i, e)}
          onKeyDown={(e) => handleKeyDown(i, e)}
          onPaste={handlePaste}
          disabled={disabled}
          className={`
            w-11 h-14 text-center text-xl font-bold rounded-xl border
            bg-slate-800/60 text-slate-100 caret-violet-400
            focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20
            disabled:opacity-40
            ${value[i] ? 'border-violet-500/60' : 'border-slate-700'}
          `}
        />
      ))}
    </div>
  );
}
