"""
Email delivery service using the Resend API.

Falls back to console logging when RESEND_API_KEY is not configured,
so the system works in development without an email provider.
"""

from __future__ import annotations

import httpx

from rag.config import get_settings
from rag.logging_config import get_logger

logger = get_logger(__name__)

_RESEND_SEND_URL = "https://api.resend.com/emails"


def send_otp_email(to_email: str, otp: str) -> bool:
    """
    Send an OTP verification email.

    Returns True on success, False on failure.
    Falls back to console output when RESEND_API_KEY is empty.
    """
    settings = get_settings()

    subject = "Your verification code"
    html_body = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:32px">
      <h2 style="color:#7c3aed;margin-bottom:8px">Verify your email</h2>
      <p style="color:#64748b;margin-bottom:24px">
        Use the code below to sign in to your RAG workspace.
      </p>
      <div style="background:#1e1b4b;border-radius:12px;padding:24px;text-align:center;margin-bottom:24px">
        <span style="font-size:36px;font-weight:700;letter-spacing:12px;color:#a78bfa">{otp}</span>
      </div>
      <p style="color:#94a3b8;font-size:13px">
        This code expires in <strong>5 minutes</strong>.<br>
        If you didn't request this, you can safely ignore this email.
      </p>
    </div>
    """

    if not settings.resend_api_key:
        # Development fallback — print to console
        logger.warning(
            "RESEND_API_KEY not set. OTP for %s: %s (console fallback)",
            to_email, otp,
        )
        print(f"\n{'='*50}\nOTP for {to_email}: {otp}\n{'='*50}\n")
        return True

    try:
        resp = httpx.post(
            _RESEND_SEND_URL,
            headers={
                "Authorization": f"Bearer {settings.resend_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": settings.email_from,
                "to": [to_email],
                "subject": subject,
                "html": html_body,
            },
            timeout=10.0,
        )
        if resp.status_code in (200, 201):
            logger.info("OTP email sent: to=%s", to_email)
            return True
        logger.error("Resend API error: status=%d body=%s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        logger.error("Email send failed: %s", exc)
        return False
