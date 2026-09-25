"""Release boundary, privacy redaction, links, ZIP metadata, and hash checks."""

import importlib.util
import json
from pathlib import Path
import stat
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_release", ROOT / "tools/public_release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def source_fixture(tmp_path):
    root = tmp_path / "repo"
    for name in release.PUBLIC_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Public source\n", encoding="utf-8")
    return root


def test_public_sources_pass():
    assert release.scan_files(ROOT, release.release_files(ROOT)) == []


@pytest.mark.parametrize("text,rule", [
    ("/" + "home" + "/" + "privateuser" + "/training", "private-filesystem-path"),
    ("/" + "root" + "/" + "work/project", "private-filesystem-path"),
    ("/" + "mnt" + "/" + "volume/models", "private-filesystem-path"),
    ("/" + "scratch" + "/" + "privateuser/checkpoints", "private-filesystem-path"),
    ("C:" + "\\" + "Users" + "\\" + "privateuser", "private-filesystem-path"),
    ("D:" + "/" + "research/models", "private-filesystem-path"),
    ("\\" * 2 + "fileserver" + "\\" + "private-share", "private-filesystem-path"),
    ("-----BEGIN " + "OPENSSH PRIVATE KEY-----", "private-key"),
    ("192." + "168.2.10", "private-ip"),
    ("10." + "2.3.4", "private-ip"),
    ("api" + "_key=" + '"' + "credential-value-123" + '"', "literal-credential"),
    ("password" + ": " + "'" + "credential-value-123" + "'", "literal-credential"),
    ("@" + "Author" + ": privateuser", "personal-author-tag"),
    ("# @" + "LastEditors" + ": privateuser", "personal-author-tag"),
    ("privateuser" + "@" + "example.org", "email-address"),
    ("ssh " + "privateuser" + "@" + "compute-node", "ssh-account"),
    ("https:" + "//" + "privateuser:password" + "@" + "example.org", "credential-in-url"),
    ("ghp_" + "a" * 36, "service-token"),
    ("hf_" + "b" * 36, "service-token"),
])
def test_detects_private_content_without_printing_it(tmp_path, text, rule):
    source = tmp_path / "README.md"
    source.write_text(text, encoding="utf-8")
    findings = release.scan_files(tmp_path, [source])
    assert any(item["rule"] == rule for item in findings)
    assert text not in json.dumps(findings)


def test_deny_token_covers_names_content_and_case_without_echo(tmp_path):
    token = "tj" + "shen"
    source = tmp_path / (token.upper() + ".py")
    source.write_text(token.upper(), encoding="utf-8")
    findings = release.scan_files(tmp_path, [source], [token])
    assert any(item["rule"] == "private-identifier" for item in findings)
    assert all(item["file"] == "<redacted-filename>" for item in findings)
    assert token not in json.dumps(findings).casefold()


def test_export_is_exactly_allowlisted_and_ignores_private_development_files(tmp_path):
    root = source_fixture(tmp_path)
    for name in (".git/config", "logs/run.log", "dapo/trainer.py", "docs/internal.md", "tests/private.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("/" + "home/privateuser/private", encoding="utf-8")
    target = tmp_path / "release"
    archive = tmp_path / "release.zip"
    result = release.export(root, target, archive=archive)
    assert result["status"] == "PASS"
    assert result["archive_verified"] is True
    assert {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()} == (
        release.PUBLIC_FILES | {release.MANIFEST}
    )
    assert release.scan_release(target)["status"] == "PASS"
    assert release.scan_release(archive)["status"] == "PASS"
    assert "privateuser" not in (target / release.MANIFEST).read_text()
    with pytest.raises(ValueError):
        release.export(root, target)
    with pytest.raises(ValueError):
        release.export(root, root / "nested")
    with pytest.raises(ValueError):
        release.export(root, tmp_path / "new", archive=archive)
    assert not (tmp_path / "new").exists()


def test_missing_allowlisted_file_fails_closed(tmp_path):
    root = source_fixture(tmp_path)
    (root / "taylormix.py").unlink()
    with pytest.raises(ValueError):
        release.export(root, tmp_path / "release")


def test_export_rejects_sensitive_source_before_writing_directory_or_archive(tmp_path):
    root = source_fixture(tmp_path)
    (root / "README.md").write_text("/" + "home" + "/" + "privateuser/data", encoding="utf-8")
    target = tmp_path / "release"
    archive = tmp_path / "release.zip"
    assert release.export(root, target, archive=archive)["status"] == "FAIL"
    assert not target.exists()
    assert not archive.exists()


@pytest.mark.parametrize("name", [
    ".git/config", "__pycache__/module.pyc", "docs/internal.md", "checkpoints/model.bin",
])
def test_directory_scan_rejects_unlisted_artifacts_without_exposing_names(tmp_path, name):
    root = source_fixture(tmp_path)
    target = tmp_path / "release"
    release.export(root, target)
    extra = target / name
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"private")
    result = release.scan_release(target)
    assert result["status"] == "FAIL"
    assert name not in json.dumps(result)


def test_manifest_detects_even_nonsensitive_content_changes(tmp_path):
    root = source_fixture(tmp_path)
    target = tmp_path / "release"
    release.export(root, target)
    (target / "README.md").write_text("Altered public source", encoding="utf-8")
    result = release.scan_release(target)
    assert result["status"] == "FAIL"
    assert any(item["rule"] == "manifest-integrity" for item in result["findings"])


def test_zip_is_deterministic_and_has_no_owner_timestamp_or_extra_metadata(tmp_path):
    root = source_fixture(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    release.export(root, tmp_path / "first", archive=first)
    release.export(root, tmp_path / "second", archive=second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.comment == b""
        for member in archive.infolist():
            assert member.date_time == release.ZIP_TIMESTAMP
            assert not member.extra and not member.comment
            assert stat.S_IFMT(member.external_attr >> 16) == stat.S_IFREG
            assert not member.filename.startswith(("/", "\\"))


@pytest.mark.parametrize("mutation", ["comment", "timestamp", "extra", "symlink", "traversal"])
def test_archive_scanner_rejects_metadata_links_and_traversal(tmp_path, mutation):
    root = source_fixture(tmp_path)
    good = tmp_path / "good.zip"
    bad = tmp_path / "bad.zip"
    release.export(root, tmp_path / "release", archive=good)
    with zipfile.ZipFile(good) as original, zipfile.ZipFile(bad, "w") as changed:
        for member in original.infolist():
            data = original.read(member)
            if member.filename == "README.md":
                if mutation == "comment":
                    member.comment = b"privateuser"
                elif mutation == "timestamp":
                    member.date_time = (2020, 1, 1, 0, 0, 0)
                elif mutation == "extra":
                    member.extra = b"\xfe\xca\x00\x00"
                elif mutation == "symlink":
                    member.external_attr = (stat.S_IFLNK | 0o777) << 16
                else:
                    member.filename = "../privateuser.txt"
            changed.writestr(member, data)
    result = release.scan_release(bad)
    assert result["status"] == "FAIL"
    assert "privateuser" not in json.dumps(result)


def test_export_rejects_symlink_source(tmp_path):
    root = source_fixture(tmp_path)
    source = tmp_path / "private.txt"
    source.write_text("private", encoding="utf-8")
    (root / "README.md").unlink()
    try:
        (root / "README.md").symlink_to(source)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    with pytest.raises(ValueError):
        release.export(root, tmp_path / "release")


def test_windows_reparse_attribute_is_rejected_without_following_target():
    class ReparsePath:
        def is_symlink(self):
            return False

        def exists(self):
            return True

        def lstat(self):
            return type("Attributes", (), {"st_file_attributes": 0x400})()

    assert release._is_link(ReparsePath())
