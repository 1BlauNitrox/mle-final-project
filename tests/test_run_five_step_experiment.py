import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts.run_five_step_experiment import board_coin_signature, process_rss


def test_board_coin_signature_accepts_numpy_coordinates_and_is_order_independent():
    field = np.array([[0, 1], [-1, 0]], dtype=np.int64)
    state = {
        "field": field,
        "coins": [(np.int64(2), np.int64(1)), (np.int64(1), np.int64(2))],
    }
    expected = hashlib.sha256(
        json.dumps([field.tolist(), [[1, 2], [2, 1]]], separators=(",", ":")).encode()
    ).hexdigest()
    assert board_coin_signature(state) == expected


def test_process_rss_accepts_supervisor_process_object():
    child = Mock()
    child.memory_info.return_value.rss = 3
    parent = Mock()
    parent.memory_info.return_value.rss = 7
    parent.children.return_value = [child]
    with patch("scripts.run_five_step_experiment.psutil.Process", return_value=parent) as factory:
        assert process_rss(SimpleNamespace(pid=123)) == 10
    factory.assert_called_once_with(123)
    parent.children.assert_called_once_with(recursive=True)
