"""Operator input stays responsive without changing physical execution."""

import math

import pytest

from tarsphere.infra.control.teleop import (
    HELP,
    HumanKeyboardHeading,
    TeleopKeyboardState,
    human_end_rotation_deg,
    human_heading_from_body_yaw,
    human_movement_heading,
    keyboard_vector,
    parse_command,
    robot_tool_request,
    speed_from_keyboard,
)


def test_combined_motion_and_release_preserve_other_held_keys():
    state = TeleopKeyboardState()
    for key in ("i", "J", "U"):
        state.press(key)
    assert keyboard_vector(state.sample(0)) == (1, 1, 1)
    state.release("J")
    assert keyboard_vector(state.sample(1)) == (1, 0, 1)
    state.press("K")
    assert keyboard_vector(state.sample(2)) == (0, 0, 1)
    state.release("K")
    assert keyboard_vector(state.sample(3)) == (1, 0, 1)
    assert speed_from_keyboard(
        state.sample(3), forward_scale=0.5, strafe_scale=0.5, yaw_scale=1
    ) == [0.5, 0, 1]


@pytest.mark.parametrize("key,direction", [("U", 1), ("O", -1)])
def test_short_turn_survives_render_batch_and_paused_simulation(key, direction):
    state = TeleopKeyboardState()
    state.press(key)
    state.release(key)
    assert keyboard_vector(state.sample(100))[2] == direction
    for _ in range(10):
        assert keyboard_vector(state.sample(100))[2] == direction
    assert keyboard_vector(state.sample(100.149))[2] == direction
    assert keyboard_vector(state.sample(100.151))[2] == 0


def test_repeat_does_not_extend_turn_pulse_and_hold_can_outlast_it():
    state = TeleopKeyboardState()
    state.press("U")
    assert keyboard_vector(state.sample(0))[2] == 1
    state.press("U")
    state.release("U")
    assert keyboard_vector(state.sample(0.16))[2] == 0
    state.press("O")
    assert keyboard_vector(state.sample(1))[2] == -1
    assert keyboard_vector(state.sample(2))[2] == -1
    state.release("O")
    assert keyboard_vector(state.sample(2.1))[2] == 0


def test_opposing_turns_cancel_and_clear_removes_pending_input():
    state = TeleopKeyboardState()
    state.press("U")
    state.press("O")
    assert keyboard_vector(state.sample(0))[2] == 0
    state.release("U")
    assert keyboard_vector(state.sample(0.01))[2] == -1
    state.release("O")
    assert keyboard_vector(state.sample(0.02))[2] == 0
    state.press("I")
    state.press("U")
    state.release("U")
    state.clear()
    assert state.sample(0.03) == [0] * 6
    assert not state.held


@pytest.mark.parametrize("duration", [0.149, -1, math.inf, math.nan])
def test_turn_minimum_is_bounded(duration):
    with pytest.raises(ValueError, match="at least 0.15"):
        TeleopKeyboardState(duration)


def test_keyboard_rejects_nonmotion_keys_and_nonfinite_clock():
    state = TeleopKeyboardState()
    with pytest.raises(ValueError, match="Unknown motion key"):
        state.press("TAB")
    with pytest.raises(ValueError, match="Unknown motion key"):
        state.release("ESCAPE")
    with pytest.raises(ValueError, match="simulation_time"):
        state.sample(math.nan)


@pytest.mark.parametrize("heading", [0, math.pi / 2, -math.pi / 2, 2.7, -3.1])
def test_visible_heading_roundtrips_through_native_root_yaw(heading):
    root_yaw = math.radians(human_end_rotation_deg(heading))
    assert human_heading_from_body_yaw(root_yaw) == pytest.approx(heading, abs=1e-5)


@pytest.mark.parametrize(
    "forward,lateral,expected",
    [
        (1, 0, 0),
        (-1, 0, math.pi),
        (0, 1, math.pi / 2),
        (0, -1, -math.pi / 2),
        (1, 1, math.pi / 4),
        (0, 0, 0),
    ],
)
def test_end_facing_uses_movement_course(forward, lateral, expected):
    assert human_movement_heading(0, forward=forward, lateral=lateral) == pytest.approx(
        expected
    )


def test_each_actor_retains_its_world_frame_and_body_follow_offset():
    actors = {
        "first": HumanKeyboardHeading(0),
        "second": HumanKeyboardHeading(math.pi / 2),
    }
    assert actors["first"].update(
        body_yaw=2, yaw_delta=0.2, follow_body=False
    ) == pytest.approx(0.2)
    assert actors["second"].yaw_rad == math.pi / 2
    assert actors["first"].update(
        body_yaw=-2, yaw_delta=0, follow_body=False
    ) == pytest.approx(0.2)
    assert actors["second"].update(
        body_yaw=math.pi / 2, yaw_delta=0.3, follow_body=True
    ) == pytest.approx(0.3)
    assert actors["second"].update(
        body_yaw=math.pi, yaw_delta=0, follow_body=True
    ) == pytest.approx(math.pi / 2 + 0.3)


def test_accepted_goto_consumes_turn_before_native_body_yaw_returns():
    heading = HumanKeyboardHeading(0)
    requested = heading.update(body_yaw=math.pi / 2, yaw_delta=0.3, follow_body=True)
    assert requested == pytest.approx(0.3)
    native_end_yaw = math.radians(human_end_rotation_deg(requested))
    heading.commit_goto()
    assert heading.yaw_rad == pytest.approx(requested)
    assert heading.body_offset_rad == 0
    assert heading.update(
        body_yaw=native_end_yaw, yaw_delta=0, follow_body=True
    ) == pytest.approx(0.3, abs=1e-5)
    # Once the explicit offset is consumed, ordinary native body turns remain
    # the source of the body-follow frame.
    assert heading.update(
        body_yaw=math.pi, yaw_delta=0, follow_body=True
    ) == pytest.approx(math.pi / 2)


def test_rejected_goto_keeps_the_unapplied_operator_turn():
    heading = HumanKeyboardHeading(0)
    heading.update(body_yaw=math.pi / 2, yaw_delta=0.3, follow_body=True)
    # A refusal leaves the native body unchanged and does not commit input.
    assert heading.update(
        body_yaw=math.pi / 2, yaw_delta=0, follow_body=True
    ) == pytest.approx(0.3)
    assert heading.body_offset_rad == pytest.approx(0.3)


def test_goto_commit_preserves_world_heading_independent_of_native_body():
    heading = HumanKeyboardHeading(0.6)
    heading.update(body_yaw=math.pi / 2, yaw_delta=0.2, follow_body=False)
    heading.commit_goto()
    assert heading.yaw_rad == pytest.approx(0.8)
    assert heading.update(body_yaw=-1, yaw_delta=0, follow_body=False) == pytest.approx(
        0.8
    )


@pytest.mark.parametrize("command", ["confirm", "confirm old-preview-token"])
def test_obsolete_confirmation_is_not_a_command(command):
    with pytest.raises(ValueError, match="Unsupported command 'confirm'"):
        parse_command(command)


def test_grasp_dispatches_directly_and_capture_is_an_operator_command():
    assert robot_tool_request(parse_command("grasp white bottle")) == (
        "grasp_object",
        {"target": "white bottle"},
    )
    command = parse_command("capture Teaser_01-front")
    assert command.kind == "capture"
    assert command.payload == {"label": "Teaser_01-front"}
    assert robot_tool_request(command) is None
    assert parse_command("capture").payload == {"label": "frame"}
    assert "confirm" not in HELP
    assert "F8" in HELP


@pytest.mark.parametrize(
    "label",
    [
        "../outside",
        "..",
        "/absolute",
        "a/b",
        "a\\b",
        "a.b",
        "汉字",
        "a" * 65,
        "two words",
    ],
)
def test_capture_rejects_unsafe_or_long_labels(label):
    with pytest.raises(ValueError, match="capture"):
        parse_command(f"capture {label}")
