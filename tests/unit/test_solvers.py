"""Test all solvers in :module:`coreax.solvers`."""
from abc import abstractmethod
from contextlib import nullcontext as does_not_raise
from typing import NamedTuple, cast
from typing_extensions import override
from unittest.mock import MagicMock
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import re

from coreax.coresets import Coreset, Coresubset
from coreax.data import Data
from coreax.util import KeyArrayLike
from coreax.kernel import Kernel, PCIMQKernel, LinearKernel, SteinKernel
from coreax.solvers import (
    Solver,
    KernelHerding,
    SteinThinning,
    RandomSample,
    RPCholesky,
)
from coreax.solvers.base import RefinementSolver, ExplicitSizeSolver
from coreax.solvers.coresubset import (
    HerdingState,
    RPCholeskyState,
    _convert_stein_kernel,
)
from coreax.score_matching import KernelDensityMatching, ScoreMatching
import pytest


class _ReduceProblem(NamedTuple):
    dataset: Data
    solver: Solver
    expected_coreset: Coreset | None = None


class _RefineProblem(NamedTuple):
    intial_coresubset: Coresubset
    solver: RefinementSolver
    expected_coresubset: Coresubset | None = None


class SolverTest:
    """Base tests for all children of :class:`coreax.solvers.Solver`."""

    random_key: KeyArrayLike = jr.key(2024)
    shape: tuple[int, int] = (100, 10)

    @abstractmethod
    def solver_factory(self) -> type[Solver] | jtu.Partial:
        """
        Pytest fixture that returns a partially applied solver initialiser.

        Partial application allows us to modify the init kwargs/args inside the tests.
        """

    @pytest.fixture
    def reduce_problem(
        self, request, solver_factory: type[Solver] | jtu.Partial
    ) -> _ReduceProblem:
        """
        Pytest fixture that returns a problem dataset and the expected coreset.

        An expected coreset of 'None' implies the expected coreset for this solver and
        dataset combination is unknown.
        """
        dataset = jr.uniform(self.random_key, self.shape)
        solver = solver_factory()
        expected_coreset = None
        return _ReduceProblem(Data(dataset), solver, expected_coreset)

    def check_solution_invariants(
        self, coreset: Coreset, problem: _RefineProblem | _ReduceProblem
    ) -> None:
        """Check that a coreset obeys certain expected invariant properties."""
        dataset, _, expected_coreset = problem
        if isinstance(problem, _RefineProblem):
            dataset = problem.intial_coresubset.pre_coreset_data
        if isinstance(coreset, Coresubset):
            assert eqx.tree_equal(coreset.pre_coreset_data, dataset)
        if expected_coreset is not None:
            assert isinstance(coreset, type(expected_coreset))
            assert eqx.tree_equal(coreset, expected_coreset)

    @pytest.mark.parametrize("use_cached_state", [False, True])
    def test_reduce(
        self, reduce_problem: _ReduceProblem, use_cached_state: bool
    ) -> None:
        """
        Check 'reduce' raises no errors and is resultant 'solver_state' invariant.

        By resultant 'solver_state' invariant we mean the following procedure succeeds:
        1. Call 'reduce' with the default 'solver_state' to get the resultant state
        2. Call 'reduce' again, this time passing the 'solver_state' from the previous
            run, and keeping all other arguments the same.
        3. Check the two calls to 'refine' yield that same result.
        """
        dataset, solver, _ = reduce_problem
        coreset, state = solver.reduce(dataset)
        if use_cached_state:
            coreset_with_state, recycled_state = solver.reduce(dataset, state)
            assert eqx.tree_equal(recycled_state, state)
            assert eqx.tree_equal(coreset_with_state, coreset)
        self.check_solution_invariants(coreset, reduce_problem)


class RefinementSolverTest(SolverTest):
    """Test cases for coresubset solvers that provide a 'refine' method."""

    @pytest.fixture(params=["wellsized", "undersized", "oversized", "random"])
    def refine_problem(self, request, reduce_problem: _ReduceProblem) -> _RefineProblem:
        """
        Pytest fixture that returns a problem dataset and the expected coreset.

        An expected coreset of 'None' implies the expected coreset for this solver and
        dataset combination is unknown.
        """
        dataset, solver, expected_coreset = reduce_problem
        indices_key, weights_key = jr.split(self.random_key)
        solver = cast(KernelHerding, solver)
        # We expect 'refine' to produce the same result as 'reduce' when the initial
        # coresubset has all its indices equal to zero.
        expected_coresubset = None
        if expected_coreset is None:
            expected_coresubset, _ = solver.reduce(dataset)
        elif isinstance(expected_coreset, Coresubset):
            expected_coresubset = expected_coreset
        if request.param == "wellsized":
            indices = Data(jnp.zeros(solver.coreset_size, jnp.int32))
        elif request.param == "undersized":
            indices = Data(jnp.zeros(solver.coreset_size - 1, jnp.int32))
        elif request.param == "oversized":
            indices = Data(jnp.zeros(solver.coreset_size + 1, jnp.int32))
        elif request.param == "random":
            random_indices = jr.choice(indices_key, solver.coreset_size, self.shape[:1])
            random_weights = jr.uniform(weights_key, self.shape[:1])
            indices = Data(random_indices, random_weights)
            expected_coresubset = None
        else:
            raise ValueError("Invalid fixture parametrization")
        initial_coresubset = Coresubset(indices, dataset)
        return _RefineProblem(initial_coresubset, solver, expected_coresubset)

    @pytest.mark.parametrize("use_cached_state", [False, True])
    def test_refine(
        self, refine_problem: _RefineProblem, use_cached_state: bool
    ) -> None:
        """
        Check 'refine' raises no errors and is resultant 'solver_state' invariant.

        By resultant 'solver_state' invariant we mean the following procedure succeeds:
        1. Call 'reduce' with the default 'solver_state' to get the inital coresubset
        2. Call 'refine' with the default 'solver_state' to get the resultant state
        3. Call 'refine' again, this time passing the 'solver_state' from the previous
            run, and keeping all other arguments the same.
        4. Check the two calls to 'refine' yield that same result.
        """
        initial_coresubset, solver, _ = refine_problem
        coresubset, state = solver.refine(initial_coresubset)
        if use_cached_state:
            coresubset_cached_state, recycled_state = solver.refine(
                initial_coresubset, state
            )
            assert eqx.tree_equal(recycled_state, state)
            assert eqx.tree_equal(coresubset_cached_state, coresubset)
        self.check_solution_invariants(coresubset, refine_problem)


class ExplicitSizeSolverTest(SolverTest):
    """Test cases for solvers that have an explicitly specified 'coreset_size'."""

    SIZE_MSG = "'coreset_size' must be a positive integer"
    OVERSIZE_MSG = "'coreset_size' must be less than 'len(dataset)'"

    @override
    def check_solution_invariants(
        self, coreset: Coreset, problem: _RefineProblem | _ReduceProblem
    ) -> None:
        super().check_solution_invariants(coreset, problem)
        solver = problem.solver
        if isinstance(solver, ExplicitSizeSolver):
            assert len(coreset) == solver.coreset_size

    @pytest.mark.parametrize(
        "coreset_size,context",
        [
            (1, does_not_raise()),
            (1.2, does_not_raise()),
            (0, pytest.raises(ValueError, match=SIZE_MSG)),
            (-1, pytest.raises(ValueError, match=SIZE_MSG)),
            ("not_castable_to_int", pytest.raises(ValueError)),
        ],
    )
    def test_check_init(
        self,
        solver_factory: type[Solver] | jtu.Partial,
        coreset_size: int | float | str,
        context,
    ) -> None:
        """
        Ensure '__check_init__' prevents initialisations with infeasible 'coreset_size'.

        A 'coreset_size' is infeasibile if it can't be cast to a positive integer.
        """
        modified_solver_factory = eqx.tree_at(
            lambda x: x.keywords["coreset_size"], solver_factory, coreset_size
        )
        with context:
            modified_solver_factory()

    def test_reduce_oversized_coreset_size(
        self, reduce_problem: _ReduceProblem
    ) -> None:
        """
        Check error is raised if 'coreset_size' is too large.

        This can not be handled in '__check_init__' as the 'coreset_size' is being
        compared against the 'dataset', which is only passed in the 'reduce' call.
        """
        dataset, solver, _ = reduce_problem
        modified_solver = eqx.tree_at(
            lambda x: x.coreset_size, solver, len(dataset) + 1
        )
        with pytest.raises(ValueError, match=re.escape(self.OVERSIZE_MSG)):
            modified_solver.reduce(dataset)


class TestKernelHerding(RefinementSolverTest, ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.KernelHerding`."""

    @override
    @pytest.fixture
    def solver_factory(self) -> jtu.Partial:
        kernel = PCIMQKernel()
        coreset_size = self.shape[0] // 10
        return jtu.Partial(KernelHerding, coreset_size=coreset_size, kernel=kernel)

    @override
    @pytest.fixture  # params=["random", "analytic"])
    def reduce_problem(
        self, request, solver_factory: type[Solver] | jtu.Partial
    ) -> _ReduceProblem:
        # if request.param == "random":
        dataset = jr.uniform(self.random_key, self.shape)
        solver = solver_factory()
        expected_coreset = None
        # elif request.param == "analytic":
        #     dataset = ...
        #     solver = ...
        #     expected_corest = ...
        # else:
        #     raise ValueError("Invalid fixture parametrization")
        return _ReduceProblem(Data(dataset), solver, expected_coreset)

    def test_herding_state(self, reduce_problem: _ReduceProblem) -> None:
        """Check that the cached herding state is as expected."""
        dataset, solver, _ = reduce_problem
        solver = cast(KernelHerding, solver)
        _, state = solver.reduce(dataset)
        expected_state = HerdingState(solver.kernel.gramian_row_mean(dataset))
        assert eqx.tree_equal(state, expected_state)


class TestRandomSample(ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.RandomSample`."""

    @override
    def check_solution_invariants(
        self, coreset: Coreset, problem: _RefineProblem | _ReduceProblem
    ) -> None:
        super().check_solution_invariants(coreset, problem)
        solver = cast(RandomSample, problem.solver)
        if solver.unique:
            _, counts = jnp.unique(coreset.nodes.data, return_counts=True)
            assert max(counts) <= 1

    @override
    @pytest.fixture
    def solver_factory(self) -> jtu.Partial:
        coreset_size = self.shape[0] // 10
        key = jr.fold_in(self.random_key, self.shape[0])
        return jtu.Partial(RandomSample, coreset_size=coreset_size, random_key=key)


class TestRPCholesky(ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.RPCholesky`."""

    @override
    def check_solution_invariants(
        self, coreset: Coreset, problem: _RefineProblem | _ReduceProblem
    ) -> None:
        super().check_solution_invariants(coreset, problem)
        solver = cast(RPCholesky, problem.solver)
        if solver.unique:
            _, counts = jnp.unique(coreset.nodes.data, return_counts=True)
            assert max(counts) <= 1

    @override
    @pytest.fixture
    def solver_factory(self) -> type[Solver] | jtu.Partial:
        kernel = PCIMQKernel()
        coreset_size = self.shape[0] // 10
        return jtu.Partial(
            RPCholesky,
            coreset_size=coreset_size,
            random_key=self.random_key,
            kernel=kernel,
        )

    def test_rpcholesky_state(self, reduce_problem: _ReduceProblem) -> None:
        """Check that the cached RPCholesky state is as expected."""
        dataset, solver, _ = reduce_problem
        solver = cast(RPCholesky, solver)
        _, state = solver.reduce(dataset)
        x = dataset.data
        gramian_diagonal = jax.vmap(solver.kernel.compute_elementwise)(x, x)
        expected_state = RPCholeskyState(gramian_diagonal)
        assert eqx.tree_equal(state, expected_state)


class TestSteinThinning(RefinementSolverTest, ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.SteinThinning`."""

    @override
    @pytest.fixture
    def solver_factory(self) -> jtu.Partial:
        kernel = PCIMQKernel()
        coreset_size = self.shape[0] // 10
        return jtu.Partial(SteinThinning, coreset_size=coreset_size, kernel=kernel)

    @pytest.mark.parametrize(
        "kernel", (PCIMQKernel(), SteinKernel(LinearKernel(), MagicMock()))
    )
    @pytest.mark.parametrize("score_matching", (None, MagicMock()))
    def test_convert_stein_kernel(
        self, kernel: Kernel, score_matching: ScoreMatching | None
    ) -> None:
        """
        Check handling of Stein kernels and standard kernels is consistent.
        """
        dataset = jr.uniform(self.random_key, self.shape)
        converted_kernel = _convert_stein_kernel(dataset, kernel, score_matching)

        if isinstance(kernel, SteinKernel):
            if score_matching is not None:
                expected_kernel = eqx.tree_at(
                    lambda x: x.score_function, kernel, score_matching.match(dataset)
                )
            expected_kernel = kernel
        else:
            if score_matching is None:
                length_scale = getattr(kernel, "length_scale", 1.0)
                score_matching = KernelDensityMatching(length_scale)
            expected_kernel = SteinKernel(
                kernel, score_function=score_matching.match(dataset)
            )
        assert eqx.tree_equal(converted_kernel.base_kernel, expected_kernel.base_kernel)
        # Score function hashes won't match; resort to checking identical evaluation.
        assert eqx.tree_equal(
            converted_kernel.score_function(dataset),
            expected_kernel.score_function(dataset),
        )
