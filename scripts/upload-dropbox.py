"""
Upload a file or directory to Dropbox with progress reporting.

Examples:

python scripts/upload-dropbox.py runs/checkpoint.pt /backups/checkpoint.pt
python scripts/upload-dropbox.py output/ /backups/output/ --overwrite
python scripts/upload-dropbox.py output/ /backups/output/ --dry-run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
from typing import Any

from tqdm import tqdm

DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_TOKEN_ENV = "DROPBOX_ACCESS_TOKEN"


def load_dropbox_sdk() -> tuple[Any, type[Exception], type[Exception], Any, Any, Any]:
    try:
        import dropbox
        from dropbox.exceptions import ApiError, AuthError
        from dropbox.files import CommitInfo, UploadSessionCursor, WriteMode
    except ImportError as exc:
        raise RuntimeError(
            "Dropbox SDK is not installed. Sync project dependencies or install dropbox manually."
        ) from exc
    return dropbox, ApiError, AuthError, CommitInfo, UploadSessionCursor, WriteMode


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def normalize_dropbox_path(path: str) -> str:
    normalized = PurePosixPath("/", path.strip() or "/").as_posix()
    return normalized if normalized != "." else "/"


def join_dropbox_path(base: str, relative_path: Path) -> str:
    path = PurePosixPath(base)
    for part in relative_path.parts:
        path /= part
    return path.as_posix()


def destination_for_file(source: Path, destination: str, destination_is_dir: bool) -> str:
    if destination_is_dir or destination == "/":
        return join_dropbox_path(destination, Path(source.name))
    return destination


def iter_upload_pairs(
    source: Path,
    destination: str,
    destination_is_dir: bool,
) -> list[tuple[Path, str]]:
    if source.is_file():
        return [(source, destination_for_file(source, destination, destination_is_dir))]

    if not source.is_dir():
        raise ValueError(f"Source path does not exist: {source}")

    files = sorted(path for path in source.rglob("*") if path.is_file())
    return [
        (path, join_dropbox_path(destination, path.relative_to(source)))
        for path in files
    ]


def resolve_token(token: str | None, token_env: str) -> str:
    if token:
        return token
    env_value = os.getenv(token_env)
    if env_value:
        return env_value
    raise ValueError(
        f"Dropbox token not provided. Use --token or set {token_env}."
    )


def ensure_dropbox_directories(dbx: Any, remote_path: str, api_error_type: type[Exception]) -> None:
    parent = PurePosixPath(remote_path).parent
    if parent.as_posix() in {".", "/"}:
        return

    current = PurePosixPath("/")
    for part in parent.parts:
        if part == "/":
            continue
        current /= part
        try:
            dbx.files_create_folder_v2(current.as_posix())
        except api_error_type as exc:
            if exc.error.is_path() and exc.error.get_path().is_conflict():
                continue
            raise


def remote_path_exists(dbx: Any, remote_path: str, api_error_type: type[Exception]) -> bool:
    try:
        dbx.files_get_metadata(remote_path)
        return True
    except api_error_type as exc:
        if exc.error.is_path() and exc.error.get_path().is_not_found():
            return False
        raise


def upload_file(
    dbx: Any,
    local_path: Path,
    remote_path: str,
    chunk_size: int,
    overwrite: bool,
    show_progress: bool,
    api_error_type: type[Exception],
    commit_info_type: Any,
    upload_session_cursor_type: Any,
    write_mode_type: Any,
) -> str:
    file_size = local_path.stat().st_size
    mode = write_mode_type.overwrite if overwrite else write_mode_type.add

    ensure_dropbox_directories(dbx, remote_path, api_error_type)

    if not overwrite and remote_path_exists(dbx, remote_path, api_error_type):
        return "skipped"

    with local_path.open("rb") as handle, tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=local_path.name,
        disable=not show_progress,
    ) as progress:
        if file_size <= chunk_size:
            payload = handle.read()
            dbx.files_upload(payload, remote_path, mode=mode, mute=True)
            progress.update(len(payload))
            return "uploaded"

        first_chunk = handle.read(chunk_size)
        session = dbx.files_upload_session_start(first_chunk)
        progress.update(len(first_chunk))

        cursor = upload_session_cursor_type(
            session_id=session.session_id,
            offset=handle.tell(),
        )
        commit = commit_info_type(path=remote_path, mode=mode, mute=True)

        while handle.tell() < file_size:
            remaining = file_size - handle.tell()
            chunk = handle.read(min(chunk_size, remaining))

            if handle.tell() == file_size:
                dbx.files_upload_session_finish(chunk, cursor, commit)
            else:
                dbx.files_upload_session_append_v2(chunk, cursor)
                cursor.offset = handle.tell()

            progress.update(len(chunk))

    return "uploaded"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload a file or directory to Dropbox")
    parser.add_argument("source", type=Path, help="local file or directory to upload")
    parser.add_argument(
        "destination",
        type=str,
        help="Dropbox destination path. For single-file uploads, a trailing / treats this as a directory.",
    )
    parser.add_argument("--token", type=str, default=None, help="Dropbox access token")
    parser.add_argument(
        "--token-env",
        type=str,
        default=DEFAULT_TOKEN_ENV,
        help=f"environment variable to read the Dropbox token from, default: {DEFAULT_TOKEN_ENV}",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=positive_int,
        default=DEFAULT_CHUNK_SIZE // (1024 * 1024),
        help="upload chunk size in MiB for large files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing files at the destination path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show planned uploads without sending anything to Dropbox",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable per-file progress bars",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.exists():
        parser.error(f"source path does not exist: {source}")

    destination = normalize_dropbox_path(args.destination)
    destination_is_dir = args.destination.endswith("/") or source.is_dir()

    upload_pairs = iter_upload_pairs(source, destination, destination_is_dir)
    if not upload_pairs:
        parser.error(f"source directory contains no files: {source}")

    total_bytes = sum(local_path.stat().st_size for local_path, _ in upload_pairs)

    if args.dry_run:
        for local_path, remote_path in upload_pairs:
            print(f"[dry-run] {local_path} -> {remote_path}")
        print(f"Planned {len(upload_pairs)} file(s), {total_bytes} bytes total.")
        return 0

    dropbox, api_error_type, auth_error_type, commit_info_type, upload_session_cursor_type, write_mode_type = load_dropbox_sdk()

    token = resolve_token(args.token, args.token_env)
    dbx = dropbox.Dropbox(token)
    dbx.users_get_current_account()

    chunk_size = args.chunk_size_mb * 1024 * 1024
    uploaded = 0
    skipped = 0

    for local_path, remote_path in upload_pairs:
        status = upload_file(
            dbx=dbx,
            local_path=local_path,
            remote_path=remote_path,
            chunk_size=chunk_size,
            overwrite=args.overwrite,
            show_progress=not args.no_progress,
            api_error_type=api_error_type,
            commit_info_type=commit_info_type,
            upload_session_cursor_type=upload_session_cursor_type,
            write_mode_type=write_mode_type,
        )
        if status == "uploaded":
            uploaded += 1
        else:
            skipped += 1
            print(f"Skipping existing file: {remote_path}")

    print(
        f"Completed upload of {uploaded} file(s); skipped {skipped} existing file(s); total bytes {total_bytes}."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OSError as exc:
        raise SystemExit(str(exc)) from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    except Exception as exc:
        name = exc.__class__.__name__
        if name == "AuthError":
            raise SystemExit(f"Dropbox authentication failed: {exc}") from exc
        if name == "ApiError":
            raise SystemExit(f"Dropbox API error: {exc}") from exc
        raise SystemExit(str(exc)) from exc

