import pytest
import torch

from scripts.audit_task3_policy_drift import compare, td_error_diagnostics


def fixture():
    state = {
        "layers.0.weight": torch.zeros(64, 39),
        "layers.0.bias": torch.ones(64),
        "layers.2.weight": torch.eye(64),
        "layers.2.bias": torch.zeros(64),
        "layers.4.weight": torch.zeros(6, 64),
        "layers.4.bias": torch.zeros(6),
    }
    state["layers.4.bias"][0] = 1
    replay = {
        "states": torch.zeros(2, 39),
        "next_states": torch.zeros(2, 39),
        "next_action_masks": torch.ones(2, 6, dtype=torch.bool),
        "rewards": torch.zeros(2),
    }
    replay["next_states"][0, 26] = 1
    return state, replay


def test_same_states_isolate_changed_greedy_action_with_mask():
    state, replay = fixture()
    learned = {k: v.clone() for k, v in state.items()}
    learned["layers.4.bias"][5] = 2
    replay["next_action_masks"][1, 5] = False
    result = compare(state, learned, replay)
    assert result["opponent_present"]["greedy_changed_fraction"] == 1
    assert result["opponent_absent"]["greedy_changed_fraction"] == 0
    assert result["all"]["final_actions"] == [1, 0, 0, 0, 0, 1]
    assert torch.equal(state["layers.4.bias"], torch.tensor([1.0, 0, 0, 0, 0, 0]))


def test_empty_and_ineligible_replay_rejected():
    state, replay = fixture()
    replay["next_action_masks"][:] = False
    with pytest.raises(ValueError, match="eligible"):
        compare(state, state, replay)
    replay["states"] = torch.zeros(0, 39)
    with pytest.raises(ValueError, match="Empty"):
        compare(state, state, replay)


def test_td_error_diagnostics_reports_direction_and_does_not_mutate_inputs():
    state, replay = fixture()
    replay["action_indices"] = torch.tensor([0, 0])
    replay["terminals"] = torch.tensor([True, False])
    learned = {key: value.clone() for key, value in state.items()}
    learned["layers.4.bias"][0] = -1
    result = td_error_diagnostics(state, state, learned, learned, replay)
    assert result["samples"] == 2
    assert result["mean_signed_td_initial"] == pytest.approx(-0.55)
    assert result["mean_signed_td_final"] == pytest.approx(1.0)
    assert result["td_sign_changed_fraction"] == pytest.approx(1)
    assert torch.equal(state["layers.4.bias"], torch.tensor([1.0, 0, 0, 0, 0, 0]))
