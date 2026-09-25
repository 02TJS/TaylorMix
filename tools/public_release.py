"""Export an exact paper-reference allowlist and inspect directory/ZIP releases."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = {
    ".gitattributes", ".gitignore", "README.md", "CODE_MANIFEST.md",
    "requirements.txt", "taylormix.py", "examples/selector_demo.py",
    "docs/PAPER_ALIGNMENT.md", "docs/INTEGRATION.md",
    "tests/test_paper_selector.py", "tests/test_public_release.py", "tools/public_release.py",
}
MANIFEST = "RELEASE_MANIFEST.json"
RELEASE_KIND = "paper-aligned-selector-reference"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MAX_FILE_BYTES = 2 * 1024 * 1024
RULES = {
    "private-filesystem-path": re.compile(
        r"(?i)(?:/(?:home|users|root|mnt|srv|zhdd|data|workspace|workspaces|scratch|"
        r"lustre|gpfs|nfs|raid|media)/[A-Za-z0-9_.-]+"
        r"|[A-Za-z]:[\\/]+[A-Za-z0-9_]|\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9_])"),
    "personal-author-tag": re.compile(r"(?im)^\s*(?:#\s*)?@(?:Author|LastEditors)\s*:"),
    "email-address": re.compile(r"\b[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "private-key": re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
    "service-token": re.compile(
        r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"
        r"|hf_[A-Za-z0-9]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{32,}"
        r"|AKIA[A-Z0-9]{16})"),
    "credential-in-url": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
    "private-ip": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}"
        r"|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"),
    "ssh-account": re.compile(r"\b(?:ssh|scp)\s+[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+"),
    "literal-credential": re.compile(
        r"""(?i)\b(?:api[_-]?key|password|passwd|access[_-]?token|secret[_-]?key|wandb[_-]?api[_-]?key)\s*[:=]\s*["'](?!EMPTY["']|YOUR_[A-Z_]+["']|["'])[^"'$\r\n]{8,}["']"""),
}


def _is_link(path):
    if path.is_symlink():
        return True
    return path.exists() and bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)


def _check_no_links(path):
    for component in (path, *path.parents):
        if _is_link(component):
            raise ValueError("Links and reparse points are not allowed.")


def _safe_name(name, deny_tokens):
    if name not in PUBLIC_FILES | {MANIFEST} or any(
        token.casefold() in name.casefold() for token in deny_tokens
    ):
        return "<redacted-filename>"
    return name


def _finding(name, rule, deny_tokens=(), line=0):
    return {"file": _safe_name(name, deny_tokens), "line": line, "rule": rule}


def _content_findings(entries, deny_tokens=()):
    if any(not token for token in deny_tokens):
        raise ValueError("Deny tokens must not be empty.")
    findings = []
    for name, data in entries.items():
        if any(token.casefold() in name.casefold() for token in deny_tokens):
            findings.append(_finding(name, "private-identifier", deny_tokens))
        try:
            text = data.decode("utf-8")
        except UnicodeError:
            findings.append(_finding(name, "non-text-artifact", deny_tokens))
            continue
        if "\x00" in text:
            findings.append(_finding(name, "non-text-artifact", deny_tokens))
        for rule, pattern in RULES.items():
            for match in pattern.finditer(text):
                findings.append(_finding(
                    name, rule, deny_tokens, text.count("\n", 0, match.start()) + 1
                ))
        for number, line in enumerate(text.splitlines(), 1):
            if any(token.casefold() in line.casefold() for token in deny_tokens):
                findings.append(_finding(name, "private-identifier", deny_tokens, number))
    return findings


def _read_file(path):
    _check_no_links(path)
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Release inputs must be small regular files.")
    return path.read_bytes()


def release_files(root):
    root = Path(root).absolute()
    _check_no_links(root)
    files = [root / name for name in sorted(PUBLIC_FILES)]
    for path in files:
        _check_no_links(path)
        if not path.is_file():
            raise ValueError("An allowlisted source file is missing.")
    return files


def scan_files(root, files, deny_tokens=()):
    entries = {path.relative_to(root).as_posix(): _read_file(path) for path in files}
    return _content_findings(entries, deny_tokens)


def _manifest_findings(entries, deny_tokens):
    findings = []
    if set(entries) != PUBLIC_FILES | {MANIFEST}:
        findings.append(_finding(MANIFEST, "release-membership", deny_tokens))
    try:
        manifest = json.loads(entries[MANIFEST])
        expected = [
            {"file": name, "sha256": hashlib.sha256(entries[name]).hexdigest()}
            for name in sorted(PUBLIC_FILES)
        ]
        if manifest != {
            "format_version": 1, "release_kind": RELEASE_KIND,
            "git_history_included": False, "files": expected,
        }:
            raise ValueError("Manifest mismatch.")
    except (KeyError, ValueError, TypeError):
        findings.append(_finding(MANIFEST, "manifest-integrity", deny_tokens))
    return findings


def scan_release(target, deny_tokens=()):
    target = Path(target).absolute()
    _check_no_links(target)
    entries = {}
    findings = []
    if target.is_dir():
        allowed_directories = {
            str(parent) for name in PUBLIC_FILES
            for parent in PurePosixPath(name).parents if str(parent) != "."
        }

        def visit(directory):
            for path in sorted(directory.iterdir()):
                name = path.relative_to(target).as_posix()
                if _is_link(path):
                    findings.append(_finding(name, "link-or-reparse-point", deny_tokens))
                elif path.is_dir():
                    if name not in allowed_directories:
                        findings.append(_finding(name, "unlisted-directory", deny_tokens))
                    else:
                        visit(path)
                elif name not in PUBLIC_FILES | {MANIFEST}:
                    findings.append(_finding(name, "unlisted-file", deny_tokens))
                else:
                    entries[name] = _read_file(path)

        visit(target)
    else:
        with zipfile.ZipFile(target) as archive:
            if archive.comment:
                findings.append(_finding("", "archive-metadata", deny_tokens))
            members = archive.infolist()
            if len(members) > len(PUBLIC_FILES) + 1:
                findings.append(_finding("", "release-membership", deny_tokens))
            for member in members:
                name = member.filename
                if name not in PUBLIC_FILES | {MANIFEST} or name in entries:
                    findings.append(_finding(name, "unlisted-or-duplicate-member", deny_tokens))
                    continue
                if (
                    member.extra or member.comment or member.date_time != ZIP_TIMESTAMP
                    or member.flag_bits & 1
                    or stat.S_IFMT(member.external_attr >> 16) != stat.S_IFREG
                ):
                    findings.append(_finding(name, "archive-metadata", deny_tokens))
                if member.file_size > MAX_FILE_BYTES:
                    findings.append(_finding(name, "oversized-member", deny_tokens))
                else:
                    entries[name] = archive.read(member)
    findings.extend(_content_findings(entries, deny_tokens))
    findings.extend(_manifest_findings(entries, deny_tokens))
    return {"status": "FAIL" if findings else "PASS", "files": len(entries), "findings": findings}


def _fresh_destination(path, root):
    path = Path(path).absolute()
    _check_no_links(path)
    if path.exists():
        raise ValueError("Output already exists; choose a fresh destination.")
    resolved = path.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError("Output must be outside the source repository.")
    return path


def export(root, destination, deny_tokens=(), archive=None):
    root = Path(root).absolute()
    _check_no_links(root)
    root = root.resolve()
    destination = _fresh_destination(destination, root)
    if archive is not None:
        archive = _fresh_destination(archive, root)
        if archive == destination or destination in archive.parents or archive in destination.parents:
            raise ValueError("Archive and release directory must be separate.")
    entries = {
        source.relative_to(root).as_posix(): _read_file(source)
        for source in release_files(root)
    }
    findings = _content_findings(entries, deny_tokens)
    if findings:
        return {"status": "FAIL", "findings": findings}
    manifest = {
        "format_version": 1, "release_kind": RELEASE_KIND, "git_history_included": False,
        "files": [
            {"file": name, "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(entries.items())
        ],
    }
    entries[MANIFEST] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    destination.mkdir(parents=True)
    for name, data in entries.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    directory_result = scan_release(destination, deny_tokens)
    if directory_result["status"] != "PASS":
        return directory_result
    if archive is not None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, data in sorted(entries.items()):
                member = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
                member.create_system = 3
                member.external_attr = (stat.S_IFREG | 0o644) << 16
                member.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(member, data)
        archive_result = scan_release(archive, deny_tokens)
        if archive_result["status"] != "PASS":
            return archive_result
    return {
        "status": "PASS", "files": len(entries), "release_kind": RELEASE_KIND,
        "git_history_included": False, "archive_verified": archive is not None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan")
    scan.add_argument("target", type=Path)
    scan.add_argument("--deny-token", action="append", default=[])
    build = commands.add_parser("export")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--archive", type=Path)
    build.add_argument("--deny-token", action="append", default=[])
    args = parser.parse_args()
    try:
        if args.command == "scan":
            result = scan_release(args.target, args.deny_token)
        else:
            result = export(SOURCE_ROOT, args.output, args.deny_token, args.archive)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile):
        result = {
            "status": "FAIL",
            "message": "Check allowlisted inputs, links, archive format, and fresh destinations.",
        }
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
