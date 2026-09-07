import math
import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "python"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from part2_control import (  # noqa: E402
    Environment,
    GMMGuidedRecedingHorizonPlanner,
    HedgeTrimmingRobot,
    MotionPlanningSimulation,
    PlannerConfig,
)


class ForwardKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.robot = HedgeTrimmingRobot()

    def assert_link_lengths(self, q):
        positions = np.asarray(self.robot.forward_kinematics(q))
        lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        expected = np.array(
            [self.robot.L1, self.robot.L2, self.robot.L3, self.robot.L4, q[3]]
        )
        np.testing.assert_allclose(lengths, expected, atol=1e-12)

    def test_zero_configuration_geometry(self):
        q = np.array([0.0, 0.0, 0.0, 0.01])
        positions = np.asarray(self.robot.forward_kinematics(q))
        expected_x = [0.0, 0.0, 0.2, 0.25, 0.45, 0.46]
        np.testing.assert_allclose(positions[:, 0], expected_x, atol=1e-12)
        np.testing.assert_allclose(positions[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            positions[:, 2], [0.0] + [self.robot.L1] * 5, atol=1e-12
        )
        self.assert_link_lengths(q)

    def test_link_lengths_are_invariant_for_nonzero_configurations(self):
        configurations = [
            np.array([0.4, 0.7, -0.3, 0.02]),
            np.array([1.0, 1.0, 0.5, 0.03]),
            np.array([-2.0, -0.8, 0.9, 0.04]),
        ]
        for q in configurations:
            with self.subTest(q=q):
                self.assert_link_lengths(q)

    def test_prismatic_motion_preserves_tool_base_and_direction(self):
        q_retracted = np.array([0.6, 0.4, -0.2, 0.0])
        q_extended = q_retracted.copy()
        q_extended[3] = 0.035
        retracted = np.asarray(self.robot.forward_kinematics(q_retracted))
        extended = np.asarray(self.robot.forward_kinematics(q_extended))
        np.testing.assert_allclose(retracted[:-1], extended[:-1], atol=1e-12)
        extension = extended[-1] - retracted[-1]
        self.assertAlmostEqual(np.linalg.norm(extension), 0.035, places=12)
        final_link = retracted[-2] - retracted[-3]
        self.assertAlmostEqual(np.linalg.norm(np.cross(extension, final_link)), 0.0, places=12)

    def test_joint_limits(self):
        self.assertTrue(self.robot.check_joint_limits([0.0, 0.0, 0.0, 0.02]))
        self.assertFalse(self.robot.check_joint_limits([0.0, 0.0, 0.0, -0.001]))
        self.assertFalse(self.robot.check_joint_limits([0.0, np.pi, 0.0, 0.02]))


class CollisionGeometryTests(unittest.TestCase):
    def setUp(self):
        self.robot = HedgeTrimmingRobot()

    def test_point_to_segment_distance(self):
        distance = Environment.point_to_segment_distance(
            np.array([0.5, 1.0, 0.0]), np.zeros(3), np.array([1.0, 0.0, 0.0])
        )
        self.assertAlmostEqual(distance, 1.0)

    def test_segment_collision_detects_clear_endpoints(self):
        obstacle = {"center": np.array([0.1, 0.0, self.robot.L1]), "radius": 0.01}
        environment = Environment([obstacle], link_safety_radius=0.005)
        positions = self.robot.forward_kinematics([0.0, 0.0, 0.0, 0.0])
        endpoint_distances = [
            np.linalg.norm(position - obstacle["center"]) for position in positions
        ]
        self.assertGreater(min(endpoint_distances), obstacle["radius"] + 0.005)
        self.assertTrue(environment.check_collision(positions))

    def test_swept_collision_detects_unsafe_midpoint(self):
        obstacle = {"center": np.array([0.2, 0.0, self.robot.L1]), "radius": 0.015}
        environment = Environment(
            [obstacle], link_safety_radius=0.002, swept_angular_resolution=math.radians(2)
        )
        q_start = np.array([-0.5, 0.0, 0.0, 0.0])
        q_end = np.array([0.5, 0.0, 0.0, 0.0])
        self.assertGreater(environment.configuration_clearance(self.robot, q_start), 0.0)
        self.assertGreater(environment.configuration_clearance(self.robot, q_end), 0.0)
        self.assertFalse(environment.is_swept_path_safe(self.robot, q_start, q_end))

    def test_adaptive_sweep_detects_previously_missed_obstacle_edge(self):
        environment = Environment()
        q_start = np.array([-0.29629455, 0.04113058, -0.19516289, 0.03166618])
        q_end = np.array([-0.34629455, -0.00886942, -0.14516289, 0.03358240])
        self.assertGreater(environment.configuration_clearance(self.robot, q_start), 0.0)
        self.assertGreater(environment.configuration_clearance(self.robot, q_end), 0.0)
        self.assertLess(
            environment.swept_path_min_clearance(self.robot, q_start, q_end), 0.0
        )
        self.assertFalse(environment.is_swept_path_safe(self.robot, q_start, q_end))

    def test_positive_clearance_tolerance_rejects_boundary_grazing(self):
        obstacle = {
            "center": np.array([0.1, 0.01525, self.robot.L1]),
            "radius": 0.01,
        }
        environment = Environment(
            [obstacle], link_safety_radius=0.005, clearance_tolerance=0.0005
        )
        q = np.array([0.0, 0.0, 0.0, 0.0])
        clearance = environment.configuration_clearance(self.robot, q)
        self.assertGreater(clearance, 0.0)
        self.assertLess(clearance, environment.clearance_tolerance)
        self.assertFalse(environment.is_swept_path_safe(self.robot, q, q))

    def test_trajectory_feasibility_checks_limits_and_swept_motion(self):
        empty_environment = Environment([], link_safety_radius=0.0)
        safe = np.array([[0.0, 0.0, 0.0, 0.01], [0.1, 0.1, 0.0, 0.02]])
        unsafe_limits = safe.copy()
        unsafe_limits[1, 3] = 0.2
        self.assertTrue(empty_environment.is_trajectory_feasible(self.robot, safe))
        self.assertFalse(empty_environment.is_trajectory_feasible(self.robot, unsafe_limits))

    def test_corrected_benchmark_scenario_endpoints_are_collision_free(self):
        environment = Environment()
        scenarios = MotionPlanningSimulation.benchmark_scenarios()
        for scenario in scenarios:
            with self.subTest(scenario=scenario.name):
                self.assertGreater(
                    environment.configuration_clearance(self.robot, scenario.q_start), 0.0
                )
                self.assertGreater(
                    environment.configuration_clearance(self.robot, scenario.q_goal), 0.0
                )
        self.assertTrue(
            environment.is_swept_path_safe(
                self.robot, scenarios[0].q_start, scenarios[0].q_goal
            )
        )
        self.assertFalse(
            environment.is_swept_path_safe(
                self.robot, scenarios[1].q_start, scenarios[1].q_goal
            )
        )


class PlannerBehaviorTests(unittest.TestCase):
    def make_planner(self, seed=11, mode="heuristic", n_training=40):
        config = PlannerConfig(
            seed=seed,
            horizon=8,
            n_candidates=12,
            n_training=n_training,
        )
        return GMMGuidedRecedingHorizonPlanner(
            HedgeTrimmingRobot(), Environment([], link_safety_radius=0.0), config, mode
        )

    def test_terminal_cost_is_active(self):
        planner = self.make_planner()
        start = np.array([0.0, 0.0, 0.0, 0.01])
        goal = np.array([1.0, 0.8, 0.5, 0.03])
        trajectory = planner._generate_straight_trajectory(start, goal)
        _, breakdown = planner.evaluate_trajectory(trajectory, goal)
        self.assertGreater(breakdown["terminal"], 0.0)
        self.assertGreater(breakdown["velocity"], 0.0)

    def test_safe_intermediate_prefers_progress_toward_goal(self):
        planner = self.make_planner(seed=3)
        start = np.array([0.0, 0.0, 0.0, 0.01])
        goal = np.array([1.0, 0.4, -0.2, 0.03])
        intermediate = planner._find_safe_intermediate_position(start, goal)
        self.assertIsNotNone(intermediate)
        before = planner.robot.normalized_distance(start, goal)
        after = planner.robot.normalized_distance(intermediate, goal)
        self.assertLess(after, before)

    def test_seeded_heuristic_sampling_is_deterministic(self):
        planner_a = self.make_planner(seed=23)
        planner_b = self.make_planner(seed=23)
        start = np.array([0.0, 0.0, 0.0, 0.01])
        goal = np.array([0.8, 0.5, 0.2, 0.03])
        samples_a = planner_a.sample_trajectories(start, goal)
        samples_b = planner_b.sample_trajectories(start, goal)
        np.testing.assert_allclose(samples_a, samples_b)

    def test_seeded_gmm_sampling_is_deterministic_and_not_repeated(self):
        planner_a = self.make_planner(seed=31, mode="gmm_heuristic", n_training=48)
        planner_b = self.make_planner(seed=31, mode="gmm_heuristic", n_training=48)
        start = np.array([0.0, 0.0, 0.0, 0.01])
        goal = np.array([0.8, 0.5, 0.2, 0.03])
        planner_a.learn_sampling_distribution(start, goal)
        planner_b.learn_sampling_distribution(start, goal)
        first_a = planner_a.sample_trajectories(start, goal)
        first_b = planner_b.sample_trajectories(start, goal)
        np.testing.assert_allclose(first_a, first_b)
        second_a = planner_a.sample_trajectories(start, goal)
        self.assertFalse(np.allclose(first_a[0], second_a[0]))

    def test_training_rng_does_not_advance_online_rng(self):
        gmm_planner = self.make_planner(seed=37, mode="gmm_heuristic", n_training=48)
        heuristic_planner = self.make_planner(seed=37, mode="heuristic", n_training=48)
        start = np.array([0.0, 0.0, 0.0, 0.01])
        goal = np.array([0.8, 0.5, 0.2, 0.03])
        gmm_planner.learn_sampling_distribution(start, goal)
        gmm_heuristic = gmm_planner._generate_heuristic_trajectory(start, goal)
        direct_heuristic = heuristic_planner._generate_heuristic_trajectory(start, goal)
        np.testing.assert_allclose(gmm_heuristic, direct_heuristic)

    def test_approximate_ik_uses_bounds_and_accuracy_threshold(self):
        planner = self.make_planner(seed=5)
        known_q = np.array([0.4, 0.2, -0.1, 0.02])
        target = planner.robot.get_end_effector_position(known_q)
        solution = planner._inverse_kinematics_approximate(target, known_q)
        self.assertIsNotNone(solution)
        self.assertTrue(planner.robot.check_joint_limits(solution))
        error = np.linalg.norm(planner.robot.get_end_effector_position(solution) - target)
        self.assertLessEqual(error, 0.01)

    def test_realized_cost_includes_stationary_execution_steps(self):
        planner = self.make_planner(seed=41)
        start = np.array([0.0, 0.0, 0.0, 0.01])
        goal = np.array([0.8, 0.5, 0.2, 0.03])
        short = np.asarray([start])
        longer = np.asarray([start, start, start])
        short_cost, _ = planner.evaluate_realized_trajectory(short, goal)
        longer_cost, _ = planner.evaluate_realized_trajectory(longer, goal)
        self.assertGreater(longer_cost, short_cost)


class SimulationMetricTests(unittest.TestCase):
    def test_success_requires_completed_dwell_and_collision_free_execution(self):
        robot = HedgeTrimmingRobot()
        obstacle = {"center": np.array([0.1, 0.0, robot.L1]), "radius": 0.02}
        environment = Environment([obstacle], link_safety_radius=0.005)
        config = PlannerConfig(seed=3, horizon=4, n_candidates=2, stable_goal_steps=2)
        simulation = MotionPlanningSimulation(
            config=config, mode="heuristic", robot=robot, environment=environment
        )
        q = np.array([0.0, 0.0, 0.0, 0.0])
        scenario = type("ScenarioLike", (), {
            "name": "colliding goal",
            "description": "test",
            "q_start": q,
            "q_goal": q,
        })()
        result = simulation.run_scenario(scenario, duration_s=0.2)
        self.assertTrue(result["reached_goal_tolerance"])
        self.assertTrue(result["completed_dwell"])
        self.assertTrue(result["collision_failure"])
        self.assertFalse(result["safe_success"])
        self.assertFalse(result["success"])

    def test_time_limit_preserves_reached_but_not_completed_dwell(self):
        config = PlannerConfig(seed=5, horizon=4, n_candidates=2, stable_goal_steps=2)
        simulation = MotionPlanningSimulation(
            config=config,
            mode="heuristic",
            environment=Environment([], link_safety_radius=0.0),
        )
        q = np.array([0.0, 0.0, 0.0, 0.01])
        scenario = type("ScenarioLike", (), {
            "name": "late arrival",
            "description": "test",
            "q_start": q,
            "q_goal": q,
        })()
        result = simulation.run_scenario(scenario, duration_s=config.dt)
        self.assertTrue(result["reached_goal_tolerance"])
        self.assertFalse(result["completed_dwell"])
        self.assertFalse(result["safe_success"])
        self.assertEqual(len(result["planning_times_s"]), 1)
        self.assertIn("planning_time_p95_s", result)
        self.assertIn("planning_deadline_miss_fraction", result)


if __name__ == "__main__":
    unittest.main()
