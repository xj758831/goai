#!/usr/bin/env python3
"""Bounded experimental WP30 recovery driven by a shadow advisory.

This module has no ROS or MuJoCo side effects.  It only converts a confirmed
``WP30_STALL_RISK`` advisory plus the current pose into a bounded body-frame
velocity command.  Callers remain responsible for explicitly labelling the
result as an experimental control takeover.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence


MARKER_1_XY = (30.0964, 15.2527)
WP30_XY = (29.5350, 16.3275)
TRIGGER_EVENT = "WP30_STALL_RISK"


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class WP30RecoveryConfig:
    minimum_base_z_m: float = 4.00
    # The stock via accepts 0.25 m.  Requiring 0.08 m made a safe low-speed
    # recenter sit in the policy's motion deadband; 0.15 m remains materially
    # stricter than the route while avoiding a higher-risk speed increase.
    marker_radius_m: float = 0.15
    official_wp30_radius_m: float = 0.18
    reverse_clear_speed: float = 0.40
    reverse_clear_duration_s: float = 2.0
    recenter_timeout_s: float = 5.0
    align_timeout_s: float = 3.0
    drive_timeout_s: float = 9.0
    total_timeout_s: float = 19.0
    progress_timeout_s: float = 4.0
    progress_epsilon_m: float = 0.03
    align_tolerance_rad: float = math.radians(10.0)
    align_stable_s: float = 0.15
    position_gain: float = 0.90
    yaw_gain: float = 1.20
    recenter_forward_limit: float = 0.24
    recenter_reverse_limit: float = 0.18
    recenter_lateral_limit: float = 0.20
    drive_forward_limit: float = 0.48
    drive_lateral_limit: float = 0.16
    yaw_limit: float = 0.60
    # Experimental v3 option.  The previous coupled drive command could keep
    # applying saturated translation while yaw diverged.  In strict mode the
    # robot aligns first, drives straight, and stops before any re-alignment.
    strict_drive_heading_gate: bool = False
    strict_drive_yaw_correction: bool = False
    drive_realign_threshold_rad: float = math.radians(7.0)
    drive_abort_threshold_rad: float = math.radians(20.0)
    max_drive_realignments: int = 2
    enable_early_progress_retry: bool = False
    early_progress_check_s: float = 2.0
    early_progress_required_m: float = 0.20
    retry_reverse_speed: float = 0.35
    retry_reverse_duration_s: float = 1.0
    max_drive_retries: int = 1
    enable_near_marker_support_recovery: bool = False
    near_marker_shell_radius_m: float = 0.18
    support_settle_duration_s: float = 0.50
    support_settle_timeout_s: float = 1.00
    fallback_recenter_timeout_s: float = 3.00
    fallback_position_gain: float = 1.35
    fallback_yaw_gain: float = 1.20
    fallback_marker_right_shift_m: float = 0.0
    fallback_marker_radius_m: float = 0.15
    fallback_marker_stable_s: float = 0.0
    fallback_marker_min_wheel_contacts: int = 0
    fallback_post_recenter_settle_duration_s: float = 0.0
    fallback_post_recenter_settle_timeout_s: float = 1.0
    fallback_align_timeout_s: float = 3.00
    fallback_align_to_wp30_heading: bool = False
    fallback_drive_ramp_duration_s: float = 0.0
    max_support_recoveries: int = 1


@dataclass(frozen=True)
class RecoveryDecision:
    command: tuple[float, float, float]
    overridden: bool
    phase: str
    reason: str


class WP30PointCloudRecovery:
    """One-shot marker-1 recenter and closed-loop WP30 entry candidate."""

    TERMINAL_PHASES = {"complete", "aborted"}

    def __init__(self, config: WP30RecoveryConfig | None = None) -> None:
        self.config = config or WP30RecoveryConfig()
        self.phase = "idle"
        self.trigger: dict[str, Any] | None = None
        self.started_at: float | None = None
        self.phase_started_at: float | None = None
        self.align_stable_since: float | None = None
        self.best_wp30_distance_m: float | None = None
        self.last_progress_at: float | None = None
        self.override_steps = 0
        self.drive_realignments = 0
        self.drive_attempts = 0
        self.drive_retries = 0
        self.drive_attempt_start_distance_m: float | None = None
        self.drive_ramp_duration_s = 0.0
        self.early_progress_checked = False
        self.support_recoveries = 0
        self.fallback_marker_stable_since: float | None = None
        self.phase_history: list[dict[str, Any]] = []
        self.final_reason: str | None = None

    @property
    def activated(self) -> bool:
        return self.started_at is not None

    @property
    def completed(self) -> bool:
        return self.phase == "complete"

    @property
    def aborted(self) -> bool:
        return self.phase == "aborted"

    @staticmethod
    def _distance(position_xyz: Sequence[float], target_xy: Sequence[float]) -> float:
        return math.hypot(
            float(target_xy[0]) - float(position_xyz[0]),
            float(target_xy[1]) - float(position_xyz[1]),
        )

    @staticmethod
    def _target_heading(position_xyz: Sequence[float], target_xy: Sequence[float]) -> float:
        return math.atan2(
            float(target_xy[1]) - float(position_xyz[1]),
            float(target_xy[0]) - float(position_xyz[0]),
        )

    def _fallback_marker(self) -> tuple[float, float]:
        shift = self.config.fallback_marker_right_shift_m
        if shift == 0.0:
            return MARKER_1_XY
        dx = WP30_XY[0] - MARKER_1_XY[0]
        dy = WP30_XY[1] - MARKER_1_XY[1]
        length = math.hypot(dx, dy)
        return (
            MARKER_1_XY[0] + shift * dy / length,
            MARKER_1_XY[1] - shift * dx / length,
        )

    def _transition(
        self,
        phase: str,
        sim_time_s: float,
        position_xyz: Sequence[float],
        reason: str,
    ) -> None:
        self.phase = phase
        self.phase_started_at = float(sim_time_s)
        self.align_stable_since = None
        self.fallback_marker_stable_since = None
        self.phase_history.append(
            {
                "phase": phase,
                "sim_time_s": float(sim_time_s),
                "base_xyz_m": [float(value) for value in position_xyz[:3]],
                "reason": reason,
            }
        )
        if phase in self.TERMINAL_PHASES:
            self.final_reason = reason

    def _activate(
        self,
        advisory: dict[str, Any],
        sim_time_s: float,
        position_xyz: Sequence[float],
    ) -> None:
        self.trigger = dict(advisory)
        self.started_at = float(sim_time_s)
        self._transition(
            "reverse_clear",
            sim_time_s,
            position_xyz,
            "confirmed WP30_STALL_RISK; retain bounded straight reverse but block strafe",
        )

    def _abort(
        self,
        sim_time_s: float,
        position_xyz: Sequence[float],
        reason: str,
    ) -> RecoveryDecision:
        self._transition("aborted", sim_time_s, position_xyz, reason)
        self.override_steps += 1
        return RecoveryDecision((0.0, 0.0, 0.0), True, self.phase, reason)

    def _body_command(
        self,
        position_xyz: Sequence[float],
        yaw_rad: float,
        target_xy: Sequence[float],
        target_heading_rad: float,
        *,
        forward_limit: float,
        reverse_limit: float,
        lateral_limit: float,
        position_gain: float | None = None,
        yaw_gain: float | None = None,
    ) -> tuple[float, float, float]:
        dx = float(target_xy[0]) - float(position_xyz[0])
        dy = float(target_xy[1]) - float(position_xyz[1])
        cosine = math.cos(yaw_rad)
        sine = math.sin(yaw_rad)
        body_forward = cosine * dx + sine * dy
        body_lateral = -sine * dx + cosine * dy
        resolved_position_gain = (
            self.config.position_gain if position_gain is None else position_gain
        )
        resolved_yaw_gain = self.config.yaw_gain if yaw_gain is None else yaw_gain
        forward = clip(
            resolved_position_gain * body_forward,
            -reverse_limit,
            forward_limit,
        )
        lateral = clip(
            resolved_position_gain * body_lateral,
            -lateral_limit,
            lateral_limit,
        )
        yaw = clip(
            resolved_yaw_gain * wrap_to_pi(target_heading_rad - yaw_rad),
            -self.config.yaw_limit,
            self.config.yaw_limit,
        )
        return forward, lateral, yaw

    def _start_drive(
        self,
        sim_time_s: float,
        position_xyz: Sequence[float],
        reason: str,
        *,
        ramp_duration_s: float = 0.0,
    ) -> None:
        self.best_wp30_distance_m = self._distance(position_xyz, WP30_XY)
        self.drive_attempt_start_distance_m = self.best_wp30_distance_m
        self.early_progress_checked = False
        self.drive_ramp_duration_s = max(0.0, float(ramp_duration_s))
        self.drive_attempts += 1
        self.last_progress_at = float(sim_time_s)
        self._transition("drive_wp30", sim_time_s, position_xyz, reason)

    def _begin_support_recovery(
        self,
        sim_time_s: float,
        position_xyz: Sequence[float],
        reason: str,
    ) -> RecoveryDecision:
        if self.support_recoveries >= self.config.max_support_recoveries:
            return self._abort(
                sim_time_s,
                position_xyz,
                "bounded near-marker support recovery already exhausted",
            )
        self.support_recoveries += 1
        self._transition("support_settle", sim_time_s, position_xyz, reason)
        self.override_steps += 1
        return RecoveryDecision(
            (0.0, 0.0, 0.0),
            True,
            self.phase,
            "stop before bounded four-wheel support settle",
        )

    def step(
        self,
        *,
        sim_time_s: float,
        position_xyz: Sequence[float],
        yaw_rad: float,
        baseline_command: Sequence[float],
        advisories: Iterable[dict[str, Any]],
        wheel_contact_count: int | None = None,
    ) -> RecoveryDecision:
        baseline = tuple(float(value) for value in baseline_command[:3])
        if not self.activated:
            advisory = next(
                (
                    item
                    for item in advisories
                    if str(item.get("event", "")) == TRIGGER_EVENT
                ),
                None,
            )
            if advisory is None:
                return RecoveryDecision(baseline, False, self.phase, "baseline")
            self._activate(advisory, sim_time_s, position_xyz)

        if self.completed:
            return RecoveryDecision(
                baseline, False, self.phase, "official WP30 radius reached; baseline released"
            )
        if self.aborted:
            self.override_steps += 1
            return RecoveryDecision(
                (0.0, 0.0, 0.0), True, self.phase, self.final_reason or "aborted"
            )

        if float(position_xyz[2]) < self.config.minimum_base_z_m:
            return self._abort(
                sim_time_s,
                position_xyz,
                f"base height below safety gate {self.config.minimum_base_z_m:.2f} m",
            )
        assert self.started_at is not None
        assert self.phase_started_at is not None
        if sim_time_s - self.started_at > self.config.total_timeout_s:
            return self._abort(
                sim_time_s, position_xyz, "candidate total timeout exceeded"
            )

        corridor_heading = self._target_heading(MARKER_1_XY, WP30_XY)
        fallback_marker = self._fallback_marker()
        fallback_corridor_heading = self._target_heading(fallback_marker, WP30_XY)
        phase_elapsed = sim_time_s - self.phase_started_at
        command: tuple[float, float, float]

        if self.phase == "reverse_clear":
            if phase_elapsed < self.config.reverse_clear_duration_s:
                self.override_steps += 1
                return RecoveryDecision(
                    (-self.config.reverse_clear_speed, 0.0, 0.0),
                    True,
                    self.phase,
                    "bounded straight reverse; no strafe",
                )
            self._transition(
                "recenter_marker_1",
                sim_time_s,
                position_xyz,
                "straight reverse complete with platform-height gate intact",
            )
            phase_elapsed = 0.0

        if self.phase == "recenter_marker_1":
            marker_distance = self._distance(position_xyz, MARKER_1_XY)
            if marker_distance <= self.config.marker_radius_m:
                self._transition(
                    "align_wp30",
                    sim_time_s,
                    position_xyz,
                    f"marker 1 recentered within {marker_distance:.3f} m",
                )
                phase_elapsed = 0.0
            elif phase_elapsed > self.config.recenter_timeout_s:
                if (
                    self.config.enable_near_marker_support_recovery
                    and marker_distance <= self.config.near_marker_shell_radius_m
                ):
                    return self._begin_support_recovery(
                        sim_time_s,
                        position_xyz,
                        (
                            f"marker recenter stalled at {marker_distance:.3f} m; "
                            "start bounded support recovery"
                        ),
                    )
                return self._abort(
                    sim_time_s, position_xyz, "marker 1 recenter timeout exceeded"
                )
            else:
                command = self._body_command(
                    position_xyz,
                    yaw_rad,
                    MARKER_1_XY,
                    corridor_heading,
                    forward_limit=self.config.recenter_forward_limit,
                    reverse_limit=self.config.recenter_reverse_limit,
                    lateral_limit=self.config.recenter_lateral_limit,
                )
                self.override_steps += 1
                return RecoveryDecision(command, True, self.phase, "recenter marker 1")

        if self.phase == "align_wp30":
            heading_error = wrap_to_pi(corridor_heading - yaw_rad)
            if abs(heading_error) <= self.config.align_tolerance_rad:
                if self.align_stable_since is None:
                    self.align_stable_since = float(sim_time_s)
                if sim_time_s - self.align_stable_since >= self.config.align_stable_s:
                    self._start_drive(
                        sim_time_s, position_xyz, "WP30 corridor heading stable"
                    )
                    phase_elapsed = 0.0
                else:
                    self.override_steps += 1
                    return RecoveryDecision(
                        (0.0, 0.0, 0.0), True, self.phase, "alignment settling"
                    )
            elif phase_elapsed > self.config.align_timeout_s:
                return self._abort(
                    sim_time_s, position_xyz, "WP30 alignment timeout exceeded"
                )
            else:
                yaw = clip(
                    self.config.yaw_gain * heading_error,
                    -self.config.yaw_limit,
                    self.config.yaw_limit,
                )
                self.override_steps += 1
                return RecoveryDecision(
                    (0.0, 0.0, yaw), True, self.phase, "align with WP30 corridor"
                )

        if self.phase == "retry_reverse_clear":
            if phase_elapsed < self.config.retry_reverse_duration_s:
                self.override_steps += 1
                return RecoveryDecision(
                    (-self.config.retry_reverse_speed, 0.0, 0.0),
                    True,
                    self.phase,
                    "bounded retry reverse; no strafe",
                )
            if self.config.enable_near_marker_support_recovery:
                return self._begin_support_recovery(
                    sim_time_s,
                    position_xyz,
                    "bounded retry reverse complete; start support recovery",
                )
            self._transition(
                "recenter_marker_1",
                sim_time_s,
                position_xyz,
                "bounded retry reverse complete; recenter marker 1",
            )
            self.override_steps += 1
            return RecoveryDecision(
                (0.0, 0.0, 0.0), True, self.phase, "stop before retry marker recenter"
            )

        if self.phase == "support_settle":
            if phase_elapsed > self.config.support_settle_timeout_s:
                return self._abort(
                    sim_time_s,
                    position_xyz,
                    "four-wheel support was not restored within bounded settle timeout",
                )
            if (
                phase_elapsed < self.config.support_settle_duration_s
                or wheel_contact_count != 4
            ):
                self.override_steps += 1
                return RecoveryDecision(
                    (0.0, 0.0, 0.0),
                    True,
                    self.phase,
                    "wait for bounded four-wheel support settle",
                )
            self._transition(
                "fallback_recenter",
                sim_time_s,
                position_xyz,
                "four-wheel support restored; use bounded higher-gain recenter",
            )
            self.override_steps += 1
            return RecoveryDecision(
                (0.0, 0.0, 0.0), True, self.phase, "stop before fallback recenter"
            )

        if self.phase == "fallback_recenter":
            marker_distance = self._distance(position_xyz, fallback_marker)
            marker_support_ok = (
                self.config.fallback_marker_min_wheel_contacts <= 0
                or (
                    wheel_contact_count is not None
                    and wheel_contact_count
                    >= self.config.fallback_marker_min_wheel_contacts
                )
            )
            marker_stable = False
            if (
                marker_distance <= self.config.fallback_marker_radius_m
                and marker_support_ok
            ):
                if self.fallback_marker_stable_since is None:
                    self.fallback_marker_stable_since = float(sim_time_s)
                marker_stable = (
                    sim_time_s - self.fallback_marker_stable_since
                    >= self.config.fallback_marker_stable_s
                )
            else:
                self.fallback_marker_stable_since = None
            if marker_stable:
                next_phase = (
                    "fallback_post_recenter_settle"
                    if self.config.fallback_post_recenter_settle_duration_s > 0.0
                    else "fallback_align"
                )
                self._transition(
                    next_phase,
                    sim_time_s,
                    position_xyz,
                    (
                        "fallback recentered marker 1 with stable support "
                        f"within {marker_distance:.3f} m"
                    ),
                )
                self.override_steps += 1
                return RecoveryDecision(
                    (0.0, 0.0, 0.0),
                    True,
                    self.phase,
                    "stop after fallback recenter",
                )
            if phase_elapsed > self.config.fallback_recenter_timeout_s:
                return self._abort(
                    sim_time_s, position_xyz, "fallback marker recenter timeout exceeded"
                )
            command = self._body_command(
                position_xyz,
                yaw_rad,
                fallback_marker,
                fallback_corridor_heading,
                forward_limit=self.config.recenter_forward_limit,
                reverse_limit=self.config.recenter_reverse_limit,
                lateral_limit=self.config.recenter_lateral_limit,
                position_gain=self.config.fallback_position_gain,
                yaw_gain=self.config.fallback_yaw_gain,
            )
            self.override_steps += 1
            return RecoveryDecision(
                command, True, self.phase, "bounded higher-gain fallback recenter"
            )

        if self.phase == "fallback_post_recenter_settle":
            if phase_elapsed > self.config.fallback_post_recenter_settle_timeout_s:
                return self._abort(
                    sim_time_s,
                    position_xyz,
                    "four-wheel support was not restored after fallback recenter",
                )
            if (
                phase_elapsed < self.config.fallback_post_recenter_settle_duration_s
                or wheel_contact_count != 4
            ):
                self.override_steps += 1
                return RecoveryDecision(
                    (0.0, 0.0, 0.0),
                    True,
                    self.phase,
                    "wait for post-recenter four-wheel support settle",
                )
            self._transition(
                "fallback_align",
                sim_time_s,
                position_xyz,
                "post-recenter four-wheel support restored",
            )
            self.override_steps += 1
            return RecoveryDecision(
                (0.0, 0.0, 0.0), True, self.phase, "stop before fallback align"
            )

        if self.phase == "fallback_align":
            fallback_heading = (
                self._target_heading(position_xyz, WP30_XY)
                if self.config.fallback_align_to_wp30_heading
                else fallback_corridor_heading
            )
            heading_error = wrap_to_pi(fallback_heading - yaw_rad)
            if abs(heading_error) <= self.config.align_tolerance_rad:
                self._start_drive(
                    sim_time_s,
                    position_xyz,
                    "fallback entered attainable WP30 heading gate",
                    ramp_duration_s=self.config.fallback_drive_ramp_duration_s,
                )
                phase_elapsed = 0.0
            elif phase_elapsed > self.config.fallback_align_timeout_s:
                return self._abort(
                    sim_time_s, position_xyz, "fallback WP30 alignment timeout exceeded"
                )
            else:
                yaw = clip(
                    self.config.fallback_yaw_gain * heading_error,
                    -self.config.yaw_limit,
                    self.config.yaw_limit,
                )
                self.override_steps += 1
                return RecoveryDecision(
                    (0.0, 0.0, yaw), True, self.phase, "fallback align with WP30 corridor"
                )

        if self.phase == "drive_wp30":
            wp30_distance = self._distance(position_xyz, WP30_XY)
            if wp30_distance <= self.config.official_wp30_radius_m:
                self._transition(
                    "complete",
                    sim_time_s,
                    position_xyz,
                    f"inside conservative WP30 radius at {wp30_distance:.3f} m",
                )
                return RecoveryDecision(
                    baseline, False, self.phase, "official WP30 radius reached"
                )
            if phase_elapsed > self.config.drive_timeout_s:
                return self._abort(
                    sim_time_s, position_xyz, "WP30 closed-loop drive timeout exceeded"
                )
            if (
                self.config.enable_early_progress_retry
                and not self.early_progress_checked
                and phase_elapsed >= self.config.early_progress_check_s
            ):
                assert self.drive_attempt_start_distance_m is not None
                early_progress = self.drive_attempt_start_distance_m - wp30_distance
                if early_progress < self.config.early_progress_required_m:
                    if self.drive_retries >= self.config.max_drive_retries:
                        return self._abort(
                            sim_time_s,
                            position_xyz,
                            "insufficient early WP30 progress after bounded retry",
                        )
                    self.drive_retries += 1
                    self._transition(
                        "retry_reverse_clear",
                        sim_time_s,
                        position_xyz,
                        (
                            f"early WP30 progress {early_progress:.3f} m below "
                            f"{self.config.early_progress_required_m:.3f} m; "
                            "start one bounded retry"
                        ),
                    )
                    self.override_steps += 1
                    return RecoveryDecision(
                        (0.0, 0.0, 0.0),
                        True,
                        self.phase,
                        "stop before bounded WP30 retry reverse",
                    )
                self.early_progress_checked = True
            if (
                self.best_wp30_distance_m is None
                or wp30_distance
                < self.best_wp30_distance_m - self.config.progress_epsilon_m
            ):
                self.best_wp30_distance_m = wp30_distance
                self.last_progress_at = float(sim_time_s)
            if (
                self.last_progress_at is not None
                and sim_time_s - self.last_progress_at > self.config.progress_timeout_s
            ):
                return self._abort(
                    sim_time_s, position_xyz, "no safe progress toward WP30"
                )
            if self.config.strict_drive_heading_gate:
                heading_error = wrap_to_pi(corridor_heading - yaw_rad)
                if abs(heading_error) > self.config.drive_abort_threshold_rad:
                    return self._abort(
                        sim_time_s,
                        position_xyz,
                        "WP30 drive heading exceeded strict abort gate",
                    )
                if abs(heading_error) > self.config.drive_realign_threshold_rad:
                    if self.drive_realignments >= self.config.max_drive_realignments:
                        return self._abort(
                            sim_time_s,
                            position_xyz,
                            "WP30 drive exhausted bounded heading re-alignments",
                        )
                    self.drive_realignments += 1
                    self._transition(
                        "align_wp30",
                        sim_time_s,
                        position_xyz,
                        "WP30 drive paused before bounded heading re-alignment",
                    )
                    self.override_steps += 1
                    return RecoveryDecision(
                        (0.0, 0.0, 0.0),
                        True,
                        self.phase,
                        "stop translation before WP30 heading re-alignment",
                    )
                dx = float(WP30_XY[0]) - float(position_xyz[0])
                dy = float(WP30_XY[1]) - float(position_xyz[1])
                forward_projection = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
                forward = clip(
                    self.config.position_gain * forward_projection,
                    0.0,
                    self.config.drive_forward_limit,
                )
                yaw = 0.0
                if self.config.strict_drive_yaw_correction:
                    yaw = clip(
                        self.config.yaw_gain * heading_error,
                        -self.config.yaw_limit,
                        self.config.yaw_limit,
                    )
                self.override_steps += 1
                return RecoveryDecision(
                    (forward, 0.0, yaw),
                    True,
                    self.phase,
                    (
                        "zero-strafe bounded-turn WP30 entry"
                        if self.config.strict_drive_yaw_correction
                        else "strict-heading straight WP30 entry"
                    ),
                )
            command = self._body_command(
                position_xyz,
                yaw_rad,
                WP30_XY,
                self._target_heading(position_xyz, WP30_XY),
                forward_limit=self.config.drive_forward_limit,
                reverse_limit=0.0,
                lateral_limit=self.config.drive_lateral_limit,
            )
            if self.drive_ramp_duration_s > 0.0:
                translation_scale = clip(
                    phase_elapsed / self.drive_ramp_duration_s, 0.0, 1.0
                )
                command = (
                    command[0] * translation_scale,
                    command[1] * translation_scale,
                    command[2],
                )
            self.override_steps += 1
            reason = (
                "ramped closed-loop WP30 entry"
                if self.drive_ramp_duration_s > 0.0
                and phase_elapsed < self.drive_ramp_duration_s
                else "closed-loop WP30 entry"
            )
            return RecoveryDecision(command, True, self.phase, reason)

        raise RuntimeError(f"unsupported WP30 recovery phase: {self.phase}")

    def summary(self) -> dict[str, Any]:
        return {
            "experimental_control_takeover": self.activated,
            "formal_baseline": False,
            "trigger_event": TRIGGER_EVENT,
            "activated": self.activated,
            "completed": self.completed,
            "aborted": self.aborted,
            "phase": self.phase,
            "final_reason": self.final_reason,
            "started_at_sim_time_s": self.started_at,
            "override_steps": self.override_steps,
            "drive_realignments": self.drive_realignments,
            "drive_attempts": self.drive_attempts,
            "drive_retries": self.drive_retries,
            "support_recoveries": self.support_recoveries,
            "trigger": self.trigger,
            "phase_history": self.phase_history,
            "config": asdict(self.config),
            "state_reset_or_teleport_used": False,
            "formal_summary_lidar_takeover_used_field_is_not_applicable": True,
        }
