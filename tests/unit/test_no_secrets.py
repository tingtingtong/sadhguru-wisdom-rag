"""Guard: no credentials in files that get committed (this repo is public)."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECRET = re.compile(
    r"gsk_[A-Za-z0-9]{20,}"
    r"|sk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{30,}"
    r"|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}"
    r"|pylf_v1_[A-Za-z0-9_]{20,}"
    r"|https://[a-z0-9-]+\.[a-z0-9-]+\.aws\.cloud\.qdrant\.io"
)
TEXT_SUFFIXES = {".py", ".md", ".toml", ".txt", ".yml", ".yaml", ".json", ".example", ".cfg", ".ini", ".html"}


def _files_that_could_be_committed() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "."],
        cwd=ROOT, capture_output=True, text=True,
    )
    return [ROOT / p for p in out.stdout.splitlines() if (ROOT / p).is_file()]


def test_no_secrets_in_committable_files():
    offenders = []
    for path in _files_that_could_be_committed():
        if path.suffix not in TEXT_SUFFIXES and path.name != ".env.example":
            continue
        if path.stat().st_size > 2_000_000:
            continue
        if SECRET.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"secret-looking values in: {offenders}"


def test_secret_files_are_gitignored():
    for name in (".env", "secrets_for_cloud.toml"):
        r = subprocess.run(["git", "check-ignore", "-q", name], cwd=ROOT)
        assert r.returncode == 0, f"{name} must be in .gitignore"
