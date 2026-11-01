"""Fail when the public tree contains excluded or unsafe release material."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".css", ".csv", ".cff", ".html", ".ini",
    ".js", ".json", ".md", ".mjs", ".ps1", ".py", ".sh", ".toml",
    ".ts", ".txt", ".yaml", ".yml",
}
SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
    "__pycache__",
}
EXCLUDED_MARKERS = (
    "".join(("f", "orge")),
    "".join(("project", "openscad")),
    "".join(("cadv", "ector")),
    "".join(("localhost:", "5050")),
)
SUSPICIOUS_NAMES = re.compile(
    r"(^|[._-])(token|secret|password|cookies?|private[-_]?key)([._-]|$)",
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
)
PRIVATE_PATH_PATTERNS = (
    re.compile(r"(?i)\b[A-Z]:\\Users\\"),
    re.compile(r"(?i)\b[A-Z]:\\[^\r\n]*Hackathon(?: data)?\\"),
    re.compile(r"(?i)(?:^|[\s'\"])\/mnt\/[a-z]\/"),
    re.compile(r"(?i)\bApp" + r"Data\\"),
    re.compile(r"(?i)(?:^|[\\/])\.(?:ssh|aws)(?:[\\/]|$)"),
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def iter_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in SKIP_PARTS for part in path.parts)
    ]


def main() -> int:
    errors: list[str] = []
    for path in iter_files():
        rel = path.relative_to(ROOT)
        rel_lower = rel.as_posix().lower()
        if path.stat().st_size >= 95 * 1024 * 1024:
            errors.append(f"oversized file: {rel}")
        if SUSPICIOUS_NAMES.search(path.name) and path.name != "SECURITY.md":
            errors.append(f"suspicious filename: {rel}")
        if any(marker in rel_lower for marker in EXCLUDED_MARKERS):
            errors.append(f"excluded external material in path: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except UnicodeError:
            errors.append(f"invalid UTF-8 text file: {rel}")
            continue
        lowered = text.lower()
        if any(marker in lowered for marker in EXCLUDED_MARKERS):
            errors.append(f"excluded external material in content: {rel}")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            errors.append(f"possible secret in content: {rel}")
        if any(pattern.search(text) for pattern in PRIVATE_PATH_PATTERNS):
            errors.append(f"private workstation path in content: {rel}")
        if path.suffix.lower() == ".md":
            for raw_target in MARKDOWN_LINK.findall(text):
                target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
                if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                local_part = unquote(target.split("#", maxsplit=1)[0])
                resolved = (path.parent / local_part).resolve()
                if not resolved.exists():
                    errors.append(f"broken relative link in {rel}: {target}")

    if errors:
        print("Public-release check failed:")
        for error in sorted(set(errors)):
            print(f"- {error}")
        return 1

    print(f"Public-release check passed for {len(iter_files())} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

