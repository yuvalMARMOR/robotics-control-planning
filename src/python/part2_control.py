"""Kinematic GMM-guided sampled receding-horizon planning for a 3R+1P arm.

The Python study is computationally independent from the legacy MATLAB dynamics
study. Its geometry is an internally consistent planner model, not a recovered
DH model or a physically validated representation of the historical mechanism.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture


Array = np.ndarray


@dataclass(frozen=True)
class Scenario:
    name: str
    q_start: Array
    q_goal: Array
    description: str


@dataclass
class PlannerConfig:
    """Numerical planner settings; command limits are not actuator ratings."""

    horizon: int = 10
    dt: float = 0.05
    n_candidates: int = 30
    n_training: int = 80
    gmm_components: int = 3
    gmm_fraction: float = 0.5
    seed: int = 42
    command_limits: Array = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 0.04], dtype=float)
    )
    clearance_soft_threshold: float = 0.05
    goal_angle_tolerance_rad: float = math.radians(3.0)
    goal_prismatic_tolerance_m: float = 0.002
    stable_goal_steps: int = 10

    def to_jsonable(self) -> dict:
        data = asdict(self)
        data["command_limits"] = self.command_limits.tolist()
        return data


class HedgeTrimmingRobot:
    """Internally consistent yaw-pitch-pitch plus prismatic planner model.

    ``L1`` is a fixed vertical base offset. ``L2`` is the horizontal shoulder
    link. ``L3`` and ``L4`` move in the radial plane selected by ``theta1``.
    The prismatic coordinate extends along the final link direction.

    The historical Python value ``L5=0.1875`` was unused and is intentionally
    absent. Available repository material is insufficient to map this model
    uniquely onto the separate MATLAB frame convention.
    """

    def __init__(self) -> None:
        self.L1 = 0.1125
        self.L2 = 0.2
        self.L3 = 0.05
        self.L4 = 0.2
        self.joint_limits = np.array(
            [
                [-np.pi, np.pi],
                [-np.pi / 2, np.pi / 2],
                [-np.pi / 2, np.pi / 2],
                [0.0, 0.04],
            ],
            dtype=float,
        )

    def forward_kinematics(self, q: Sequence[float]) -> list[Array]:
        """Return base, joint, tool-base, and end-effector positions in metres."""
        theta1, theta2, theta3, d4 = np.asarray(q, dtype=float)
        radial = np.array([np.cos(theta1), np.sin(theta1), 0.0])
        vertical = np.array([0.0, 0.0, 1.0])

        p0 = np.zeros(3)
        p1 = p0 + self.L1 * vertical
        p2 = p1 + self.L2 * radial

        direction_3 = np.cos(theta2) * radial + np.sin(theta2) * vertical
        p3 = p2 + self.L3 * direction_3

        final_pitch = theta2 + theta3
        direction_4 = np.cos(final_pitch) * radial + np.sin(final_pitch) * vertical
        p4 = p3 + self.L4 * direction_4
        p_end = p4 + d4 * direction_4
        return [p0, p1, p2, p3, p4, p_end]

    def get_end_effector_position(self, q: Sequence[float]) -> Array:
        return self.forward_kinematics(q)[-1]

    def check_joint_limits(self, q: Sequence[float], atol: float = 1e-12) -> bool:
        q_array = np.asarray(q, dtype=float)
        return bool(
            np.all(q_array >= self.joint_limits[:, 0] - atol)
            and np.all(q_array <= self.joint_limits[:, 1] + atol)
        )

    def clip_to_joint_limits(self, q: Sequence[float]) -> Array:
        q_array = np.asarray(q, dtype=float)
        return np.clip(q_array, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def normalized_distance(self, q_a: Sequence[float], q_b: Sequence[float]) -> float:
        ranges = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        return float(np.linalg.norm((np.asarray(q_a) - np.asarray(q_b)) / ranges))


class Environment:
    """Spherical obstacles with segment-based robot clearance queries.

    Swept edges are checked by adaptive configuration-space subdivision.  The
    procedure is deliberately described as discretized rather than analytic
    continuous collision detection.  A positive clearance tolerance keeps
    numerically boundary-touching motions out of the feasible set.
    """

    def __init__(
        self,
        obstacles: Sequence[dict] | None = None,
        link_safety_radius: float = 0.02,
        clearance_tolerance: float = 0.0005,
        swept_angular_resolution: float = math.radians(0.25),
        swept_prismatic_resolution: float = 0.00025,
    ) -> None:
        self.obstacles = list(obstacles) if obstacles is not None else [
            {"center": np.array([0.15, 0.15, 0.25]), "radius": 0.08},
            {"center": np.array([-0.1, 0.2, 0.15]), "radius": 0.06},
            {"center": np.array([0.25, -0.1, 0.2]), "radius": 0.07},
        ]
        self._obstacle_centers = np.asarray(
            [obstacle["center"] for obstacle in self.obstacles], dtype=float
        ).reshape((-1, 3))
        self._obstacle_radii = np.asarray(
            [obstacle["radius"] for obstacle in self.obstacles], dtype=float
        )
        self.link_safety_radius = float(link_safety_radius)
        self.clearance_tolerance = float(clearance_tolerance)
        self.swept_angular_resolution = float(swept_angular_resolution)
        self.swept_prismatic_resolution = float(swept_prismatic_resolution)

    def to_jsonable(self) -> dict:
        return {
            "obstacles": [
                {"center": np.asarray(item["center"]).tolist(), "radius": item["radius"]}
                for item in self.obstacles
            ],
            "link_safety_radius_m": self.link_safety_radius,
            "clearance_tolerance_m": self.clearance_tolerance,
            "adaptive_angular_resolution_rad": self.swept_angular_resolution,
            "adaptive_prismatic_resolution_m": self.swept_prismatic_resolution,
            "checker": "adaptive discretized configuration-space edge subdivision",
        }

    @staticmethod
    def point_to_segment_distance(point: Array, start: Array, end: Array) -> float:
        segment = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length_sq = float(segment @ segment)
        if length_sq <= 1e-18:
            return float(np.linalg.norm(np.asarray(point) - np.asarray(start)))
        alpha = float(np.dot(np.asarray(point) - np.asarray(start), segment) / length_sq)
        closest = np.asarray(start) + np.clip(alpha, 0.0, 1.0) * segment
        return float(np.linalg.norm(np.asarray(point) - closest))

    def minimum_link_clearance(self, positions: Sequence[Array]) -> float:
        """Signed clearance: positive is safe, zero is contact, negative collides."""
        if len(self.obstacles) == 0:
            return math.inf
        points = np.asarray(positions, dtype=float)
        starts = points[:-1]
        segments = points[1:] - starts
        length_sq = np.sum(segments**2, axis=1)
        safe_length_sq = np.where(length_sq > 1e-18, length_sq, 1.0)
        offsets = self._obstacle_centers[None, :, :] - starts[:, None, :]
        alpha = np.sum(offsets * segments[:, None, :], axis=2) / safe_length_sq[:, None]
        alpha = np.clip(alpha, 0.0, 1.0)
        closest = starts[:, None, :] + alpha[:, :, None] * segments[:, None, :]
        distances = np.linalg.norm(self._obstacle_centers[None, :, :] - closest, axis=2)
        clearances = distances - (self._obstacle_radii[None, :] + self.link_safety_radius)
        return float(np.min(clearances))

    def configuration_clearance(self, robot: HedgeTrimmingRobot, q: Sequence[float]) -> float:
        return self.minimum_link_clearance(robot.forward_kinematics(q))

    def check_collision(self, positions: Sequence[Array]) -> bool:
        """Return whether any complete link segment intersects a safety-inflated sphere."""
        return self.minimum_link_clearance(positions) <= 0.0

    @staticmethod
    def _configuration_motion_bound(robot: HedgeTrimmingRobot, delta: Array) -> float:
        """Conservative workspace-displacement scale for one configuration interval."""
        delta = np.abs(np.asarray(delta, dtype=float))
        maximum_extension = robot.joint_limits[3, 1]
        yaw_reach = robot.L2 + robot.L3 + robot.L4 + maximum_extension
        shoulder_reach = robot.L3 + robot.L4 + maximum_extension
        elbow_reach = robot.L4 + maximum_extension
        return float(
            yaw_reach * delta[0]
            + shoulder_reach * delta[1]
            + elbow_reach * delta[2]
            + delta[3]
        )

    def _edge_is_at_resolution(self, q_start: Array, q_end: Array) -> bool:
        delta = np.abs(np.asarray(q_end) - np.asarray(q_start))
        return bool(
            np.max(delta[:3]) <= self.swept_angular_resolution
            and delta[3] <= self.swept_prismatic_resolution
        )

    def swept_path_min_clearance(
        self, robot: HedgeTrimmingRobot, q_start: Sequence[float], q_end: Sequence[float]
    ) -> float:
        """Adaptive discretized minimum along a linear configuration-space edge."""
        q0 = np.asarray(q_start, dtype=float)
        q1 = np.asarray(q_end, dtype=float)
        clearance0 = self.configuration_clearance(robot, q0)
        clearance1 = self.configuration_clearance(robot, q1)

        def recurse(
            left: Array,
            right: Array,
            left_clearance: float,
            right_clearance: float,
        ) -> float:
            midpoint = 0.5 * (left + right)
            midpoint_clearance = self.configuration_clearance(robot, midpoint)
            sampled_minimum = min(left_clearance, midpoint_clearance, right_clearance)
            if sampled_minimum <= self.clearance_tolerance:
                return float(sampled_minimum)
            if self._edge_is_at_resolution(left, right):
                return float(sampled_minimum)

            # Every point in either half-interval lies within one quarter of the
            # full joint interval from an endpoint or the midpoint.  When the
            # sampled clearance exceeds this workspace-motion scale plus the
            # acceptance tolerance, further subdivision cannot affect the hard
            # decision under this conservative kinematic bound.
            quarter_motion_bound = self._configuration_motion_bound(
                robot, 0.25 * (right - left)
            )
            if sampled_minimum - self.clearance_tolerance > quarter_motion_bound:
                return float(sampled_minimum)

            left_minimum = recurse(
                left, midpoint, left_clearance, midpoint_clearance
            )
            if left_minimum <= self.clearance_tolerance:
                return float(left_minimum)
            right_minimum = recurse(
                midpoint, right, midpoint_clearance, right_clearance
            )
            return float(min(left_minimum, right_minimum))

        return recurse(q0, q1, clearance0, clearance1)

    def is_swept_path_safe(
        self, robot: HedgeTrimmingRobot, q_start: Sequence[float], q_end: Sequence[float]
    ) -> bool:
        return (
            self.swept_path_min_clearance(robot, q_start, q_end)
            > self.clearance_tolerance
        )

    def trajectory_min_clearance(self, robot: HedgeTrimmingRobot, trajectory: Array) -> float:
        if len(trajectory) == 0:
            return math.inf
        if len(trajectory) == 1:
            return self.configuration_clearance(robot, trajectory[0])
        return float(
            min(
                self.swept_path_min_clearance(robot, q_start, q_end)
                for q_start, q_end in zip(trajectory[:-1], trajectory[1:])
            )
        )

    def is_trajectory_feasible(self, robot: HedgeTrimmingRobot, trajectory: Array) -> bool:
        if len(trajectory) == 0:
            return False
        if not all(robot.check_joint_limits(q) for q in trajectory):
            return False
        return (
            self.trajectory_min_clearance(robot, trajectory)
            > self.clearance_tolerance
        )


class GMMGuidedRecedingHorizonPlanner:
    """Sample and rank finite-horizon kinematic trajectories.

    The GMM is fitted separately for one scenario. It learns residuals around a
    step-limited straight trajectory; it is not conditioned on an environment
    representation and does not claim generalization to unseen planning tasks.
    """

    VALID_MODES = {"gmm_heuristic", "heuristic", "uniform"}

    def __init__(
        self,
        robot: HedgeTrimmingRobot,
        environment: Environment,
        config: PlannerConfig | None = None,
        mode: str = "gmm_heuristic",
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown sampling mode: {mode}")
        self.robot = robot
        self.env = environment
        self.config = config or PlannerConfig()
        self.mode = mode
        training_seed, online_seed = np.random.SeedSequence(self.config.seed).spawn(2)
        self.training_rng = np.random.default_rng(training_seed)
        self.online_rng = np.random.default_rng(online_seed)
        self.learned_distribution: GaussianMixture | None = None
        self.training_start: Array | None = None
        self.training_goal: Array | None = None
        self.last_diagnostics: dict = {}

        self.Q_position = np.diag([8.0, 8.0, 8.0, 120.0])
        self.Q_velocity = np.diag([0.02, 0.02, 0.02, 0.2])
        self.Q_terminal = np.diag([40.0, 40.0, 40.0, 600.0])
        self.task_position_weight = 40.0
        self.clearance_weight = 120.0

    @property
    def maximum_step(self) -> Array:
        return np.asarray(self.config.command_limits, dtype=float) * self.config.dt

    def _step_toward(self, q: Array, target: Array) -> Array:
        step = np.clip(np.asarray(target) - np.asarray(q), -self.maximum_step, self.maximum_step)
        return self.robot.clip_to_joint_limits(np.asarray(q) + step)

    def _generate_straight_trajectory(self, q_start: Array, q_goal: Array) -> Array:
        trajectory = np.zeros((self.config.horizon, 4), dtype=float)
        trajectory[0] = q_start
        for index in range(1, self.config.horizon):
            trajectory[index] = self._step_toward(trajectory[index - 1], q_goal)
        return trajectory

    def _sample_intermediate_configuration(
        self, q_start: Array, q_goal: Array, rng: np.random.Generator | None = None
    ) -> Array:
        rng = rng or self.online_rng
        for _ in range(30):
            alpha = rng.uniform(0.25, 0.75)
            line_point = (1.0 - alpha) * q_start + alpha * q_goal
            ranges = self.robot.joint_limits[:, 1] - self.robot.joint_limits[:, 0]
            candidate = line_point + rng.normal(0.0, 0.08, 4) * ranges
            candidate = self.robot.clip_to_joint_limits(candidate)
            if (
                self.env.configuration_clearance(self.robot, candidate)
                > self.env.clearance_tolerance
            ):
                return candidate
        return self.robot.clip_to_joint_limits(0.5 * (q_start + q_goal))

    def _generate_via_point_trajectory(
        self, q_start: Array, q_mid: Array, q_goal: Array
    ) -> Array:
        trajectory = np.zeros((self.config.horizon, 4), dtype=float)
        trajectory[0] = q_start
        switch = max(2, self.config.horizon // 2)
        for index in range(1, self.config.horizon):
            target = q_mid if index < switch else q_goal
            trajectory[index] = self._step_toward(trajectory[index - 1], target)
        return trajectory

    def _generate_random_trajectory(
        self, q_start: Array, q_goal: Array, rng: np.random.Generator | None = None
    ) -> Array:
        rng = rng or self.online_rng
        trajectory = np.zeros((self.config.horizon, 4), dtype=float)
        trajectory[0] = q_start
        ranges = self.robot.joint_limits[:, 1] - self.robot.joint_limits[:, 0]
        for index in range(1, self.config.horizon):
            progress = index / (self.config.horizon - 1)
            nominal = (1.0 - progress) * q_start + progress * q_goal
            noise_scale = 0.06 * (1.0 - progress)
            target = nominal + rng.normal(0.0, noise_scale, 4) * ranges
            target = self.robot.clip_to_joint_limits(target)
            trajectory[index] = self._step_toward(trajectory[index - 1], target)
        return trajectory

    def _inverse_kinematics_approximate(
        self,
        target_position: Array,
        q_initial: Array,
        cartesian_tolerance_m: float = 0.01,
    ) -> Array | None:
        """Bounded numerical IK with a 10 mm default Cartesian acceptance threshold."""

        def objective(q: Array) -> float:
            error = self.robot.get_end_effector_position(q) - target_position
            clearance = self.env.configuration_clearance(self.robot, q)
            collision_penalty = 0.0 if clearance > 0.0 else 1e3 + 1e3 * clearance**2
            return float(error @ error + collision_penalty)

        bounds = [tuple(limit) for limit in self.robot.joint_limits]
        result = minimize(
            objective,
            self.robot.clip_to_joint_limits(q_initial),
            method="Powell",
            bounds=bounds,
            options={"maxiter": 250, "xtol": 1e-6, "ftol": 1e-10},
        )
        candidate = self.robot.clip_to_joint_limits(result.x)
        error = np.linalg.norm(self.robot.get_end_effector_position(candidate) - target_position)
        if (
            result.success
            and error <= cartesian_tolerance_m
            and self.env.configuration_clearance(self.robot, candidate)
            > self.env.clearance_tolerance
        ):
            return candidate
        return None

    def _generate_obstacle_avoiding_trajectory(
        self, q_start: Array, q_goal: Array, rng: np.random.Generator | None = None
    ) -> Array:
        """Construct a feasible local rollout by ranking safe one-step proposals."""
        rng = rng or self.online_rng
        trajectory = np.zeros((self.config.horizon, 4), dtype=float)
        trajectory[0] = q_start
        lower = self.robot.joint_limits[:, 0]
        upper = self.robot.joint_limits[:, 1]
        for index in range(1, self.config.horizon):
            previous = trajectory[index - 1]
            targets = [q_goal]
            targets.extend(rng.uniform(lower, upper) for _ in range(6))
            best_next = previous.copy()
            best_score = math.inf
            for target in targets:
                proposed = self._step_toward(previous, np.asarray(target))
                if not self.env.is_swept_path_safe(self.robot, previous, proposed):
                    continue
                remaining = self.robot.normalized_distance(proposed, q_goal)
                clearance = self.env.configuration_clearance(self.robot, proposed)
                score = remaining - 0.08 * min(clearance, 0.1)
                if score < best_score:
                    best_score = score
                    best_next = proposed
            trajectory[index] = best_next
        return trajectory

    def _generate_heuristic_trajectory(
        self, q_start: Array, q_goal: Array, rng: np.random.Generator | None = None
    ) -> Array:
        rng = rng or self.online_rng
        draw = rng.random()
        if draw < 0.25:
            return self._generate_straight_trajectory(q_start, q_goal)
        if draw < 0.50:
            q_mid = self._sample_intermediate_configuration(q_start, q_goal, rng)
            return self._generate_via_point_trajectory(q_start, q_mid, q_goal)
        if draw < 0.75:
            return self._generate_random_trajectory(q_start, q_goal, rng)
        return self._generate_obstacle_avoiding_trajectory(q_start, q_goal, rng)

    def _generate_uniform_trajectory(
        self, q_start: Array, rng: np.random.Generator | None = None
    ) -> Array:
        rng = rng or self.online_rng
        trajectory = np.zeros((self.config.horizon, 4), dtype=float)
        trajectory[0] = q_start
        lower = self.robot.joint_limits[:, 0]
        upper = self.robot.joint_limits[:, 1]
        for index in range(1, self.config.horizon):
            target = rng.uniform(lower, upper)
            trajectory[index] = self._step_toward(trajectory[index - 1], target)
        return trajectory

    def learn_sampling_distribution(self, q_start: Array, q_goal: Array) -> dict:
        """Fit one scenario-specific GMM over residual trajectory states 1..N-1."""
        start_time = time.perf_counter()
        baseline = self._generate_straight_trajectory(q_start, q_goal)
        residual_samples: list[Array] = []
        generators = (
            lambda: self._generate_straight_trajectory(q_start, q_goal),
            lambda: self._generate_via_point_trajectory(
                q_start,
                self._sample_intermediate_configuration(q_start, q_goal, self.training_rng),
                q_goal,
            ),
            lambda: self._generate_random_trajectory(q_start, q_goal, self.training_rng),
            lambda: self._generate_obstacle_avoiding_trajectory(
                q_start, q_goal, self.training_rng
            ),
        )
        for index in range(self.config.n_training):
            trajectory = generators[index % len(generators)]()
            if self.env.is_trajectory_feasible(self.robot, trajectory):
                residual_samples.append((trajectory[1:] - baseline[1:]).reshape(-1))

        self.learned_distribution = None
        if len(residual_samples) >= 10:
            components = min(self.config.gmm_components, max(1, len(residual_samples) // 10))
            fit_seed = int(self.training_rng.integers(0, np.iinfo(np.int32).max))
            self.learned_distribution = GaussianMixture(
                n_components=components,
                covariance_type="full",
                random_state=fit_seed,
                reg_covar=1e-6,
            ).fit(np.asarray(residual_samples))
        self.training_start = np.asarray(q_start, dtype=float).copy()
        self.training_goal = np.asarray(q_goal, dtype=float).copy()
        return {
            "attempted": self.config.n_training,
            "feasible": len(residual_samples),
            "fitted": self.learned_distribution is not None,
            "training_wall_time_s": time.perf_counter() - start_time,
        }

    def _sample_gmm_residuals(self, count: int) -> Array:
        if self.learned_distribution is None or count <= 0:
            return np.empty((0, (self.config.horizon - 1) * 4))
        gmm = self.learned_distribution
        components = self.online_rng.choice(len(gmm.weights_), size=count, p=gmm.weights_)
        samples = np.empty((count, gmm.means_.shape[1]), dtype=float)
        for index, component in enumerate(components):
            samples[index] = self.online_rng.multivariate_normal(
                gmm.means_[component], gmm.covariances_[component]
            )
        return samples

    def sample_trajectories(self, q_start: Array, q_goal: Array) -> list[Array]:
        trajectories: list[Array] = []
        total = self.config.n_candidates
        if self.mode == "gmm_heuristic" and self.learned_distribution is not None:
            learned_count = int(round(total * self.config.gmm_fraction))
            baseline = self._generate_straight_trajectory(q_start, q_goal)
            for residual in self._sample_gmm_residuals(learned_count):
                trajectory = baseline.copy()
                trajectory[1:] += residual.reshape(self.config.horizon - 1, 4)
                for index in range(1, self.config.horizon):
                    proposed = self.robot.clip_to_joint_limits(trajectory[index])
                    trajectory[index] = self._step_toward(trajectory[index - 1], proposed)
                trajectories.append(trajectory)
            while len(trajectories) < total:
                trajectories.append(
                    self._generate_heuristic_trajectory(q_start, q_goal, self.online_rng)
                )
        elif self.mode == "heuristic" or (
            self.mode == "gmm_heuristic" and self.learned_distribution is None
        ):
            trajectories = [
                self._generate_heuristic_trajectory(q_start, q_goal, self.online_rng)
                for _ in range(total)
            ]
        else:
            trajectories = [
                self._generate_uniform_trajectory(q_start, self.online_rng)
                for _ in range(total)
            ]
        return trajectories

    def evaluate_trajectory(self, trajectory: Array, q_goal: Array) -> tuple[float, dict]:
        if len(trajectory) == 0 or not all(
            self.robot.check_joint_limits(q) for q in trajectory
        ):
            return math.inf, {"feasible": False}

        swept_clearances = [
            self.env.configuration_clearance(self.robot, trajectory[0])
        ]
        swept_clearances.extend(
            self.env.swept_path_min_clearance(self.robot, q_start, q_end)
            for q_start, q_end in zip(trajectory[:-1], trajectory[1:])
        )
        if min(swept_clearances) <= self.env.clearance_tolerance:
            return math.inf, {"feasible": False}

        position_cost = 0.0
        task_cost = 0.0
        velocity_cost = 0.0
        clearance_cost = 0.0
        goal_position = self.robot.get_end_effector_position(q_goal)

        for index, q in enumerate(trajectory):
            phase_weight = 1.0 + 2.0 * index / max(1, len(trajectory) - 1)
            error = q - q_goal
            position_cost += phase_weight * float(error @ self.Q_position @ error) * self.config.dt

            end_error = self.robot.get_end_effector_position(q) - goal_position
            task_cost += (
                phase_weight * self.task_position_weight * float(end_error @ end_error) * self.config.dt
            )

            clearance = swept_clearances[index]
            deficit = max(0.0, self.config.clearance_soft_threshold - clearance)
            clearance_cost += self.clearance_weight * deficit**2 * self.config.dt

            if index > 0:
                velocity = (trajectory[index] - trajectory[index - 1]) / self.config.dt
                velocity_cost += float(velocity @ self.Q_velocity @ velocity) * self.config.dt

        terminal_error = trajectory[-1] - q_goal
        terminal_cost = float(terminal_error @ self.Q_terminal @ terminal_error)
        total = position_cost + task_cost + velocity_cost + clearance_cost + terminal_cost
        return total, {
            "feasible": True,
            "position": position_cost,
            "task": task_cost,
            "velocity": velocity_cost,
            "clearance": clearance_cost,
            "terminal": terminal_cost,
            "total": total,
        }

    def evaluate_realized_trajectory(self, trajectory: Array, q_goal: Array) -> tuple[float, dict]:
        """Evaluate every actually executed state and edge with one common cost.

        Unlike the online finite-horizon ranking cost, this cumulative metric has
        no feasibility rejection and therefore includes fallback and stationary
        steps.  Position, task-space, velocity, and swept-clearance terms are
        integrated over the executed duration, followed by one terminal penalty.
        """
        if len(trajectory) == 0:
            return math.inf, {"total": math.inf}

        position_cost = 0.0
        task_cost = 0.0
        velocity_cost = 0.0
        clearance_cost = 0.0
        goal_position = self.robot.get_end_effector_position(q_goal)

        for index, q in enumerate(trajectory):
            error = q - q_goal
            position_cost += float(error @ self.Q_position @ error) * self.config.dt
            end_error = self.robot.get_end_effector_position(q) - goal_position
            task_cost += self.task_position_weight * float(end_error @ end_error) * self.config.dt

            if index == 0:
                clearance = self.env.configuration_clearance(self.robot, q)
            else:
                clearance = self.env.swept_path_min_clearance(
                    self.robot, trajectory[index - 1], q
                )
                velocity = (q - trajectory[index - 1]) / self.config.dt
                velocity_cost += float(velocity @ self.Q_velocity @ velocity) * self.config.dt
            deficit = max(0.0, self.config.clearance_soft_threshold - clearance)
            clearance_cost += self.clearance_weight * deficit**2 * self.config.dt

        terminal_error = trajectory[-1] - q_goal
        terminal_cost = float(terminal_error @ self.Q_terminal @ terminal_error)
        total = position_cost + task_cost + velocity_cost + clearance_cost + terminal_cost
        return total, {
            "position": position_cost,
            "task": task_cost,
            "velocity": velocity_cost,
            "clearance": clearance_cost,
            "terminal": terminal_cost,
            "total": total,
        }

    def _find_safe_intermediate_position(self, q_current: Array, q_goal: Array) -> Array | None:
        ranges = self.robot.joint_limits[:, 1] - self.robot.joint_limits[:, 0]
        current_scaled = (q_current - self.robot.joint_limits[:, 0]) / ranges
        goal_scaled = (q_goal - self.robot.joint_limits[:, 0]) / ranges
        direction = goal_scaled - current_scaled
        original_distance = np.linalg.norm(direction)
        if original_distance <= 1e-12:
            return q_current.copy()
        direction /= original_distance

        best_q = None
        best_remaining_ratio = math.inf
        for step in (0.04, 0.08, 0.12):
            directions = [direction]
            for _ in range(8):
                random_vector = self.online_rng.normal(size=4)
                orthogonal = random_vector - float(random_vector @ direction) * direction
                norm = np.linalg.norm(orthogonal)
                if norm > 1e-12:
                    directions.append(direction + 0.35 * orthogonal / norm)
            for trial_direction in directions:
                trial_direction = trial_direction / np.linalg.norm(trial_direction)
                q_scaled = np.clip(current_scaled + step * trial_direction, 0.0, 1.0)
                q_test = self.robot.joint_limits[:, 0] + q_scaled * ranges
                if not self.env.is_swept_path_safe(self.robot, q_current, q_test):
                    continue
                remaining_ratio = np.linalg.norm(goal_scaled - q_scaled) / original_distance
                if remaining_ratio < best_remaining_ratio:
                    best_remaining_ratio = remaining_ratio
                    best_q = q_test
        return best_q

    def _safe_p_fallback(self, q_current: Array, q_goal: Array) -> Array:
        direct_target = self._step_toward(q_current, q_goal)
        if self.env.is_swept_path_safe(self.robot, q_current, direct_target):
            return (direct_target - q_current) / self.config.dt
        intermediate = self._find_safe_intermediate_position(q_current, q_goal)
        if intermediate is not None:
            target = self._step_toward(q_current, intermediate)
            if self.env.is_swept_path_safe(self.robot, q_current, target):
                return (target - q_current) / self.config.dt
        return np.zeros(4)

    def compute_control(self, q_current: Array, q_goal: Array) -> Array:
        candidates = self.sample_trajectories(q_current, q_goal)
        best_trajectory = None
        best_cost = math.inf
        best_breakdown: dict = {}
        feasible_count = 0
        for trajectory in candidates:
            cost, breakdown = self.evaluate_trajectory(trajectory, q_goal)
            if math.isfinite(cost):
                feasible_count += 1
            if cost < best_cost:
                best_cost = cost
                best_trajectory = trajectory
                best_breakdown = breakdown

        used_fallback = best_trajectory is None
        if best_trajectory is not None:
            command = (best_trajectory[1] - q_current) / self.config.dt
            command = np.clip(command, -self.config.command_limits, self.config.command_limits)
            q_next = self.robot.clip_to_joint_limits(q_current + command * self.config.dt)
            if not self.env.is_swept_path_safe(self.robot, q_current, q_next):
                used_fallback = True
                command = self._safe_p_fallback(q_current, q_goal)
        else:
            command = self._safe_p_fallback(q_current, q_goal)

        self.last_diagnostics = {
            "candidate_count": len(candidates),
            "feasible_candidate_count": feasible_count,
            "selected_cost": best_cost if math.isfinite(best_cost) else None,
            "cost_breakdown": best_breakdown,
            "used_p_fallback": used_fallback,
        }
        return command


class MotionPlanningSimulation:
    """Execute the independent kinematic planner and collect reproducible metrics."""

    def __init__(
        self,
        config: PlannerConfig | None = None,
        mode: str = "gmm_heuristic",
        robot: HedgeTrimmingRobot | None = None,
        environment: Environment | None = None,
    ) -> None:
        self.config = config or PlannerConfig()
        self.robot = robot or HedgeTrimmingRobot()
        self.environment = environment or Environment()
        self.planner = GMMGuidedRecedingHorizonPlanner(
            self.robot, self.environment, self.config, mode=mode
        )
        self.mode = mode

    @staticmethod
    def historical_scenarios() -> list[Scenario]:
        """Original scenario definitions retained for provenance.

        Under the corrected segment collision model, two historical goals are
        inside safety-inflated obstacles. They are therefore not valid benchmark
        queries for the corrected planner.
        """
        return [
            Scenario(
                "Point-to-Point",
                np.array([0.0, 0.0, 0.0, 0.01]),
                np.array([np.pi / 3, np.pi / 4, np.pi / 6, 0.03]),
                "Simple point-to-point motion",
            ),
            Scenario(
                "Obstacle Avoidance",
                np.array([-np.pi / 4, -np.pi / 6, 0.0, 0.01]),
                np.array([np.pi / 4, np.pi / 3, np.pi / 4, 0.035]),
                "Motion planning around spherical obstacles",
            ),
            Scenario(
                "Complex Maneuvering",
                np.array([np.pi / 2, -np.pi / 4, -np.pi / 6, 0.005]),
                np.array([-np.pi / 3, np.pi / 3, np.pi / 3, 0.038]),
                "Complex maneuvering through constrained workspace regions",
            ),
        ]

    @staticmethod
    def benchmark_scenarios() -> list[Scenario]:
        """Synthetic, endpoint-valid queries for the corrected kinematic model."""
        return [
            Scenario(
                "Validated Point-to-Point",
                np.array([0.0, 0.0, 0.0, 0.01]),
                np.array([0.6, -0.4, 0.2, 0.025]),
                "Collision-free direct query for the corrected Python geometry",
            ),
            Scenario(
                "Validated Obstacle Detour",
                np.array([0.0, 0.0, 0.0, 0.01]),
                np.array([-0.6, 0.25, -0.3, 0.03]),
                "Safe endpoints with a colliding direct interpolation",
            ),
            Scenario(
                "Validated Multi-Axis Motion",
                np.array([np.pi / 2, -np.pi / 4, -np.pi / 6, 0.005]),
                np.array([0.6, -0.4, 0.2, 0.025]),
                "Collision-free multi-axis query from the historical complex start",
            ),
        ]

    @staticmethod
    def scenarios() -> list[Scenario]:
        """Return synthetic endpoint-checked scenarios used by demos and benchmarks."""
        return MotionPlanningSimulation.benchmark_scenarios()

    def run_scenario(self, scenario: Scenario, duration_s: float = 6.0) -> dict:
        query_start_time = time.perf_counter()
        training = {
            "attempted": 0,
            "feasible": 0,
            "fitted": False,
            "training_wall_time_s": 0.0,
        }
        if self.mode == "gmm_heuristic":
            training = self.planner.learn_sampling_distribution(
                scenario.q_start, scenario.q_goal
            )

        q_current = scenario.q_start.copy()
        trajectory = [q_current.copy()]
        times = [0.0]
        planning_times: list[float] = []
        selected_costs: list[float] = []
        fallback_steps = 0
        no_motion_steps = 0
        stable_steps = 0
        reached_goal_tolerance = False
        completed_dwell = False
        executed_collision = False

        for step in range(int(round(duration_s / self.config.dt))):
            start = time.perf_counter()
            command = self.planner.compute_control(q_current, scenario.q_goal)
            planning_times.append(time.perf_counter() - start)
            diagnostics = self.planner.last_diagnostics
            if diagnostics.get("selected_cost") is not None:
                selected_costs.append(float(diagnostics["selected_cost"]))
            fallback_steps += int(bool(diagnostics.get("used_p_fallback")))

            q_next = self.robot.clip_to_joint_limits(q_current + command * self.config.dt)
            clearance = self.environment.swept_path_min_clearance(
                self.robot, q_current, q_next
            )
            if clearance <= self.environment.clearance_tolerance:
                executed_collision = True
                q_next = q_current.copy()
            if np.allclose(q_next, q_current, atol=1e-12):
                no_motion_steps += 1
            q_current = q_next
            trajectory.append(q_current.copy())
            times.append((step + 1) * self.config.dt)

            error = np.abs(q_current - scenario.q_goal)
            at_goal = bool(
                np.all(error[:3] <= self.config.goal_angle_tolerance_rad)
                and error[3] <= self.config.goal_prismatic_tolerance_m
            )
            reached_goal_tolerance = reached_goal_tolerance or at_goal
            stable_steps = stable_steps + 1 if at_goal else 0
            if stable_steps >= self.config.stable_goal_steps:
                completed_dwell = True
                break

        query_solver_wall_time = time.perf_counter() - query_start_time

        trajectory_array = np.asarray(trajectory)
        end_positions = np.asarray(
            [self.robot.get_end_effector_position(q) for q in trajectory_array]
        )
        path_length = float(np.sum(np.linalg.norm(np.diff(end_positions, axis=0), axis=1)))
        final_delta = q_current - scenario.q_goal
        final_end_error = float(
            np.linalg.norm(
                self.robot.get_end_effector_position(q_current)
                - self.robot.get_end_effector_position(scenario.q_goal)
            )
        )
        minimum_clearance = self.environment.trajectory_min_clearance(
            self.robot, trajectory_array
        )
        collision_failure = bool(
            executed_collision
            or minimum_clearance <= self.environment.clearance_tolerance
        )
        safe_success = bool(completed_dwell and not collision_failure)
        realized_cost, realized_breakdown = self.planner.evaluate_realized_trajectory(
            trajectory_array, scenario.q_goal
        )
        normalized_config_error = self.robot.normalized_distance(q_current, scenario.q_goal)
        planning_time_array = np.asarray(planning_times, dtype=float)
        return {
            "scenario": scenario.name,
            "description": scenario.description,
            "mode": self.mode,
            "seed": self.config.seed,
            "control_period_s": self.config.dt,
            "success": safe_success,
            "goal_reached": completed_dwell,
            "reached_goal_tolerance": reached_goal_tolerance,
            "completed_dwell": completed_dwell,
            "safe_success": safe_success,
            "collision_failure": collision_failure,
            "q_start": scenario.q_start.tolist(),
            "q_goal": scenario.q_goal.tolist(),
            "q_final": q_current.tolist(),
            "final_joint_error": final_delta.tolist(),
            "final_angular_error_l2_rad": float(np.linalg.norm(final_delta[:3])),
            "final_prismatic_error_m": float(abs(final_delta[3])),
            "normalized_configuration_error": normalized_config_error,
            "end_effector_error_m": final_end_error,
            "end_effector_path_length_m": path_length,
            "minimum_link_clearance_m": minimum_clearance,
            "mean_selected_horizon_cost": float(np.mean(selected_costs)) if selected_costs else None,
            "realized_trajectory_cost": realized_cost,
            "realized_cost_breakdown": realized_breakdown,
            "planning_time_mean_s": float(np.mean(planning_times)) if planning_times else 0.0,
            "planning_time_median_s": float(np.median(planning_times)) if planning_times else 0.0,
            "planning_time_p95_s": (
                float(np.quantile(planning_time_array, 0.95)) if planning_times else 0.0
            ),
            "planning_time_max_s": float(np.max(planning_times)) if planning_times else 0.0,
            "planning_time_total_s": float(np.sum(planning_times)),
            "planning_deadline_miss_fraction": (
                float(np.mean(planning_time_array > self.config.dt)) if planning_times else 0.0
            ),
            "planning_times_s": planning_times,
            "query_solver_wall_time_s": query_solver_wall_time,
            "fitting_plus_online_planning_time_s": (
                training["training_wall_time_s"] + float(np.sum(planning_times))
            ),
            "execution_duration_s": times[-1],
            "control_steps": len(times) - 1,
            "fallback_steps": fallback_steps,
            "no_motion_steps": no_motion_steps,
            "training": training,
            "times_s": times,
            "trajectory": trajectory_array.tolist(),
            "end_effector_trajectory": end_positions.tolist(),
        }


def dependency_versions() -> dict:
    packages = ["numpy", "matplotlib", "scipy", "scikit-learn", "Pillow"]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def save_scenario_plots(result: dict, environment: Environment, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    times = np.asarray(result["times_s"])
    q = np.asarray(result["trajectory"])
    q_goal = np.asarray(result["q_goal"])

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    labels = [r"$\theta_1$ (rad)", r"$\theta_2$ (rad)", r"$\theta_3$ (rad)", r"$d_4$ (m)"]
    for index, axis in enumerate(axes.flat):
        axis.plot(times, q[:, index], label="executed")
        axis.axhline(q_goal[index], color="tab:red", linestyle="--", label="goal")
        axis.set_ylabel(labels[index])
        axis.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 1].set_xlabel("time (s)")
    axes[0, 0].legend()
    figure.suptitle(f"{result['scenario']} — {result['mode']}, seed {result['seed']}")
    figure.tight_layout()
    stem = result["scenario"].lower().replace(" ", "-")
    figure.savefig(output_directory / f"{stem}-configuration.png", dpi=180)
    plt.close(figure)

    end_positions = np.asarray(result["end_effector_trajectory"])
    robot = HedgeTrimmingRobot()
    goal_position = robot.get_end_effector_position(np.asarray(result["q_goal"]))
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(end_positions[:, 0], end_positions[:, 1], end_positions[:, 2], label="end effector")
    axis.scatter(*end_positions[0], color="tab:green", s=60, label="start")
    axis.scatter(*goal_position, color="tab:red", s=90, marker="*", label="goal")
    axis.scatter(*end_positions[-1], color="black", s=45, marker="x", label="final")
    sphere_u = np.linspace(0.0, 2.0 * np.pi, 28)
    sphere_v = np.linspace(0.0, np.pi, 16)
    for obstacle in environment.obstacles:
        center = np.asarray(obstacle["center"])
        radius = float(obstacle["radius"])
        x = center[0] + radius * np.outer(np.cos(sphere_u), np.sin(sphere_v))
        y = center[1] + radius * np.outer(np.sin(sphere_u), np.sin(sphere_v))
        z = center[2] + radius * np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
        axis.plot_surface(x, y, z, color="tab:red", alpha=0.18, linewidth=0)
        safety_radius = radius + environment.link_safety_radius
        x_safe = center[0] + safety_radius * np.outer(np.cos(sphere_u), np.sin(sphere_v))
        y_safe = center[1] + safety_radius * np.outer(np.sin(sphere_u), np.sin(sphere_v))
        z_safe = center[2] + safety_radius * np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
        axis.plot_wireframe(
            x_safe, y_safe, z_safe, color="tab:orange", alpha=0.18, linewidth=0.35
        )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title(
        f"Minimum swept post-radius clearance: {result['minimum_link_clearance_m']:.4f} m"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / f"{stem}-task-space.png", dpi=180)
    plt.close(figure)


def run_demo(output_directory: Path, seed: int = 42) -> list[dict]:
    config = PlannerConfig(seed=seed)
    results = []
    for scenario in MotionPlanningSimulation.scenarios():
        simulation = MotionPlanningSimulation(config=config, mode="gmm_heuristic")
        result = simulation.run_scenario(scenario)
        results.append(result)
        save_scenario_plots(result, simulation.environment, output_directory)
    payload = {
        "method": "GMM-guided trajectory sampling with MPC-style sampled receding-horizon selection",
        "scope": "Independent kinematic Python planning study",
        "config": config.to_jsonable(),
        "environment": dependency_versions(),
        "results": results,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "demo-results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the corrected GMM-guided sampled receding-horizon planning demo."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("results") / "corrected" / "demo"
    )
    arguments = parser.parse_args()
    results = run_demo(arguments.output, seed=arguments.seed)
    for result in results:
        print(
            f"{result['scenario']}: safe_success={result['safe_success']}, "
            f"end_error={result['end_effector_error_m']:.4f} m, "
            f"min_clearance={result['minimum_link_clearance_m']:.4f} m"
        )


if __name__ == "__main__":
    main()
