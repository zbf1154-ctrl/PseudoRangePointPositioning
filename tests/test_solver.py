import numpy as np
import pytest

from spp.config import SolverConfig
from spp.solver import SppSolver


def test_solve_short_window(dataset, short_result):
    assert short_result.solved_epochs == 11
    assert short_result.rejected_epochs == 0
    assert short_result.solutions[0].time.hour == 0
    assert short_result.solutions[0].time.minute == 0
    assert short_result.mode_counts() == {"dual": 11}

    offsets = short_result.offsets_ecef(use_truth=True)
    mean_error = float(np.mean(np.linalg.norm(offsets, axis=1)))
    assert mean_error < 50.0
    assert np.mean([solution.residual_rms for solution in short_result.solutions]) < 20.0
    assert np.all([solution.satellites >= 4 for solution in short_result.solutions])


def test_first_epoch_near_clock_boundary(dataset):
    solver = SppSolver(dataset)
    solution = solver.solve_epoch(dataset.observations[0])
    assert solution is not None
    assert np.all(np.isfinite(solution.position))
    assert solution.clock_bias_s != 0.0


def test_single_frequency_mode(dataset):
    solver = SppSolver(dataset, SolverConfig(frequency_mode="single"))
    solutions = solver.solve()
    assert len(solutions) == len(dataset.observations)
    assert all(solution.mode == "single" for solution in solutions)


def test_without_troposphere_still_converges(dataset):
    solver = SppSolver(
        dataset,
        SolverConfig(frequency_mode="single", use_troposphere=False, robust=False),
    )
    solutions = solver.solve()
    assert len(solutions) == len(dataset.observations)
    assert all(np.isfinite(solution.position).all() for solution in solutions)


def test_solution_clock_matches_station_clock(dataset, short_result):
    station_clock = dataset.clocks.station_bias("ABMF")
    assert station_clock is not None
    first = short_result.solutions[0]
    assert first.clock_bias_s == pytest.approx(station_clock, abs=1e-6)
