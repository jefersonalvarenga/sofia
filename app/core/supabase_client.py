"""
Supabase client singleton for Sofia.
"""

from typing import Optional
from supabase import create_client, Client


_client: Optional[Client] = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        from app.core.config import get_settings
        settings = get_settings()
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client
