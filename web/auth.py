import os
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

COOKIE_NAME = "pipeline_session"
COOKIE_MAX_AGE = 86400 * 7


class AuthManager:
    def __init__(self, password: str, secret_key: str | None = None):
        self._password = password
        self._secret = secret_key or os.urandom(32).hex()
        self._serializer = URLSafeTimedSerializer(self._secret)

    def check_password(self, password: str) -> bool:
        return password == self._password

    def create_session(self) -> str:
        return self._serializer.dumps({"authenticated": True})

    def validate_session(self, token: str) -> bool:
        try:
            data = self._serializer.loads(token, max_age=COOKIE_MAX_AGE)
            return data.get("authenticated", False)
        except (BadSignature, SignatureExpired):
            return False


def load_password_from_env() -> str:
    from pathlib import Path
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass
    pw = os.environ.get("WEB_PASSWORD")
    if not pw:
        print("ERROR: WEB_PASSWORD not set", flush=True)
        raise SystemExit(1)
    return pw
