import os
import sys
import json
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional


class SimpleLogger:
    """
    A JSONL logger that temporarily moves the institution log file to /tmp,
    appends the new entry, and moves it back to the source directory.
    """

    def __init__(self, log_path: str, institution_id: Optional[str] = None):
        self._institution_id = institution_id
        self._final_log_path = log_path

        tmp_dir = "/tmp/logs"
        os.makedirs(tmp_dir, exist_ok=True)

        # Use a temp file per institution to avoid collision
        self._tmp_log_path = os.path.join(
            tmp_dir, f"{institution_id}_validation.tmp.log"
        )

        # If final log exists, move it into tmp before writing
        if os.path.exists(self._final_log_path):
            shutil.copy2(self._final_log_path, self._tmp_log_path)
        else:
            # Create empty temp log file
            open(self._tmp_log_path, "w", encoding="utf-8").close()

        # Open for appending
        self._fh = open(self._tmp_log_path, "a", encoding="utf-8")

    def _write(self, entry: Dict[str, Any]) -> None:
        entry.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
        line = json.dumps(entry, default=str, indent=4)
        self._fh.write(line + "\n")
        self._fh.flush()
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def error(
        self,
        message: Optional[str] = None,
        *,
        extra_columns: Optional[List[str]] = None,
        missing_required: Optional[List[str]] = None,
        schema_errors: Any = None,
        failure_cases: Any = None,
    ) -> None:
        entry: Dict[str, Any] = {"validation_status": "hard_error"}
        if message:
            entry["message"] = message
        if extra_columns is not None:
            entry["extra_columns"] = extra_columns
        if missing_required is not None:
            entry["missing_required"] = missing_required
        if schema_errors is not None:
            entry["schema_errors"] = schema_errors
        if failure_cases is not None:
            entry["failure_cases"] = failure_cases
        self._write(entry)

    def info(self, *, missing_optional: Optional[List[str]] = None) -> None:
        status = "passed_with_soft_errors" if missing_optional else "passed"
        entry = {
            "validation_status": status,
            "missing_optional": missing_optional or [],
        }
        self._write(entry)

    def exception(self, message: str = "Unexpected exception occurred") -> None:
        exc_type, exc_val, _ = sys.exc_info()
        entry = {
            "validation_status": "exception",
            "message": message,
            "error_type": exc_type.__name__ if exc_type else None,
            "error": str(exc_val),
        }
        self._write(entry)

    def close(self) -> None:
        try:
            self._fh.close()
            final_dir = os.path.dirname(self._final_log_path)
            os.makedirs(final_dir, exist_ok=True)
            shutil.move(self._tmp_log_path, self._final_log_path)
        except Exception as e:
            sys.stderr.write(f"[LOGGER] Failed to move final log file: {e}\n")


def setup_logger(
    institution_id: Optional[str] = None, log_file: str = "validation.log"
) -> SimpleLogger:
    if not institution_id:
        raise ValueError("institution_id is required for institution-specific logging")

    log_dir = f"/Volumes/staging_sst_01/{institution_id}_bronze/bronze_volume/logs"
    log_path = os.path.join(log_dir, log_file)

    return SimpleLogger(log_path=log_path, institution_id=institution_id)
