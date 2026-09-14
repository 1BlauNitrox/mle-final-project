"""Bounded storage-only workaround for issue175 Windows progress-file sharing errors."""

import json
import time
from pathlib import Path


def with_permission_retry(write, audit, *, retries=100, delay=0.05):
    """Retry the identical serialized write; never suppress permanent failures."""

    def retry(path, value):
        for attempt in range(retries + 1):
            try:
                return write(path, value)
            except PermissionError as error:
                if attempt == retries:
                    raise
                with Path(audit).open("a", encoding="utf-8") as record:
                    record.write(
                        json.dumps(
                            {
                                "time": time.time(),
                                "path": str(path),
                                "retry": attempt + 1,
                                "error": str(error),
                            }
                        )
                        + "\n"
                    )
                time.sleep(delay)

    return retry
