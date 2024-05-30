# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for refinement implementations.

Refinement approaches greedily select points to improve coreset quality. The tests
within this file verify that refinement approaches used produce the expected results on
simple examples.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import NamedTuple
from unittest.mock import patch

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array

import coreax.approximation
import coreax.data
import coreax.kernel
import coreax.reduction
import coreax.refine
import coreax.util


class CoresetMock(coreax.reduction.Coreset):
    """Test version of :class:`Coreset` with all methods implemented."""

    def fit_to_size(self, coreset_size: int):
        raise NotImplementedError


class TestRefine:
    """Tests related to `coreax.refine.Refine`."""

    random_key = jr.key(0)

    # Disable pylint warning for protected-access as we are testing a single part of
    # the over-arching algorithm ``coreax.refine.Refine._validate_coreset``.
    # pylint: disable=protected-access
    def test_validate_coreset_ok(self):
        """Check validation passes with populated coresubset."""
        coreset = CoresetMock()
        coreset.original_data = coreax.data.ArrayData.load(1)
        coreset.coreset = jnp.array(1)
        coreset.coreset_indices = jnp.array(0)
        coreax.refine.Refine._validate_coreset(coreset)

    def test_validate_coreset_no_fit(self):
        """Check validation fails when coreset has not been calculated."""
        coreset = CoresetMock()
        coreset.original_data = coreax.data.ArrayData.load(1)
        with pytest.raises(coreax.util.NotCalculatedError):
            coreax.refine.Refine._validate_coreset(coreset)

    def test_validate_coreset_not_coresubset(self):
        """Check validation raises TypeError when not a coresubset."""
        coreset = CoresetMock()
        coreset.original_data = coreax.data.ArrayData.load(1)
        coreset.coreset = jnp.array(1)

        with pytest.raises(TypeError):
            coreax.refine.Refine._validate_coreset(coreset)

    # pylint: enable=protected-access


class _RefineProblem(NamedTuple):
    array: Array
    test_indices: list[tuple[int, ...] | list[int] | set[int]]
    best_indices: list[set[int]]


class RefineTestCase(ABC):
    """Abstract test case for concrete implementations of `coreax.refine.Refine`."""

    random_key = jr.key(0)

    @abstractmethod
    def refine_method(self) -> coreax.refine.Refine:
        """Abstract pytest fixture returning an initialised refine object."""

    def test_invalid_kernel_argument(self, refine_method: coreax.refine.Refine):
        """Test behaviour of the ``refine`` method when passed an invalid kernel."""
        original_array = jnp.asarray([[0, 0], [1, 1], [0, 0], [1, 1]])
        best_indices = {0, 1}
        coreset_indices = jnp.array(list(best_indices))
        coreset_obj = CoresetMock(weights_optimiser=None, kernel=None)
        coreset_obj.coreset_indices = coreset_indices
        coreset_obj.original_data = coreax.data.ArrayData.load(original_array)
        coreset_obj.coreset = original_array[coreset_indices, :]

        # Attempt to refine using this coreset - we don't have a kernel, and so we
        # should attempt to call compute on a None type object, which should raise an
        # attribute error
        with pytest.raises(AttributeError, match="object has no attribute 'compute'"):
            refine_method.refine(coreset=coreset_obj)

    def test_invalid_coreset_argument(self, refine_method: coreax.refine.Refine):
        """Test behaviour of the ``refine`` method when passed an invalid coreset."""
        # Define an object to pass that is not a coreset, and does not have the
        # associated attributes required to refine
        with pytest.raises(
            AttributeError, match="object has no attribute 'validate_fitted'"
        ):
            refine_method.refine(coreset=coreax.util.InvalidKernel(x=1.0))

    @pytest.mark.parametrize(
        "problem",
        [
            _RefineProblem(
                array=jnp.asarray([[0, 0], [1, 1], [0, 0], [1, 1]]),
                test_indices=list(itertools.combinations(range(4), 2)),
                best_indices=[{0, 1}, {0, 3}, {1, 2}, {2, 3}],
            ),
            _RefineProblem(
                array=jnp.asarray([[0, 0], [1, 1], [2, 2]]),
                test_indices=[(2, 2)],
                best_indices=[{0, 2}],
            ),
        ],
        ids=["no_unique_best", "unique_best"],
    )
    @pytest.mark.parametrize("use_cached_row_mean", [False, True])
    def test_refine(
        self,
        refine_method: coreax.refine.Refine,
        problem: _RefineProblem,
        use_cached_row_mean: bool,
    ):
        """
        Test behaviour of the ``refine`` method when passed valid arguments.

        Each test problem consists of an ``array``, a list of sets of ``test_indices``,
        and a list of sets of ``best_indices``. We test that, when given a coresubset
        on ``array`` with indices given by any set in ``test_indices``, the ``refine``
        method returns a coresubset with indices equal to a set in ``best_indices``.

        - The ``no_unique_best`` case tests scenarios with multiple "best" solutions.
        - The ``unique_best`` case tests scenarios with a unique "best" solution. This
        unique solution is expected even for random/greedy refinement methods.

        When ``use_cached_row_mean=True``, we expect the corresponding cached value in
        the coreset object to be used by refine, otherwise, we expect the kernel's
        gramian_row_mean to be called (exactly once).
        """
        array, test_indices, best_indices = problem
        for indices in test_indices:
            coreset_indices = jnp.array(indices)
            kernel = coreax.kernel.SquaredExponentialKernel()
            coreset_obj = CoresetMock(
                weights_optimiser=None,
                kernel=kernel,
            )
            coreset_obj.coreset_indices = coreset_indices
            coreset_obj.original_data = coreax.data.ArrayData.load(array)
            coreset_obj.coreset = array[coreset_indices, :]

            if use_cached_row_mean:
                refine_method.refine(coreset=coreset_obj)
            else:
                # If we aren't using the cached gramian_row_mean, then
                # gramian_row_mean should be called exactly once.
                coreset_obj.gramian_row_mean = None
                with patch.object(
                    coreax.kernel.Kernel,
                    "gramian_row_mean",
                    wraps=kernel.gramian_row_mean,
                ) as mock_method:
                    refine_method.refine(coreset=coreset_obj)
                mock_method.assert_called_once()
                refine_method.refine(coreset=coreset_obj)
            assert any(
                set(coreset_obj.coreset_indices.tolist()) == i for i in best_indices
            )


class TestRefineRegular(RefineTestCase):
    """Tests related to :meth:`~coreax.refine.RefineRegular`."""

    @pytest.fixture
    def refine_method(self):
        return coreax.refine.RefineRegular()


class TestRefineRandom(RefineTestCase):
    """Tests related to :meth:`~coreax.refine.RefineRandom`."""

    @pytest.fixture
    def refine_method(self):
        return coreax.refine.RefineRandom(self.random_key, p=1.0)

    def test_zero_original_data_points(self):
        """
        Test how RefineRandom handles a coreset with no original data to refine with.
        """
        original_array = jnp.asarray([])
        coreset_indices = jnp.array([])
        kernel = coreax.kernel.SquaredExponentialKernel()
        coreset_obj = CoresetMock(weights_optimiser=None, kernel=kernel)
        coreset_obj.coreset_indices = coreset_indices
        coreset_obj.original_data = coreax.data.ArrayData.load(original_array)
        coreset_obj.coreset = jnp.asarray([])
        coreset_obj.gramian_row_mean = None

        # Attempt to refine a coreset by considering no original data points
        # which should try to divide by 0 and raise a value error highlighting
        # the root cause
        refine_random = coreax.refine.RefineRandom(self.random_key, p=0.5)
        with pytest.raises(ValueError, match="must not be empty"):
            refine_random.refine(coreset=coreset_obj)

    @pytest.mark.parametrize("p", [0, -0.5, 1.5])
    def test_original_data_proportion(self, p: float):
        """
        Test RefineRandom on different (degenerate) original data sampling proportions.

        ``p`` should be capped at ``1``, and an error raised for ``p <= 0``.
        """
        original_array = jnp.asarray([[0, 0], [1, 1], [2, 2]])
        test_indices = [2, 2]
        coreset_indices = jnp.array(test_indices)
        kernel = coreax.kernel.SquaredExponentialKernel()
        coreset_obj = CoresetMock(weights_optimiser=None, kernel=kernel)
        coreset_obj.coreset_indices = coreset_indices
        coreset_obj.original_data = coreax.data.ArrayData.load(original_array)
        coreset_obj.coreset = original_array[coreset_indices, :]
        coreset_obj.gramian_row_mean = None

        refine_random = coreax.refine.RefineRandom(self.random_key, p=p)
        if p <= 0:
            with pytest.raises(ValueError, match="input p must be greater than 0"):
                refine_random.refine(coreset=coreset_obj)
        else:
            refine_random.refine(coreset=coreset_obj)
            assert refine_random.p == min(p, 1.0)


class TestRefineReverse(RefineTestCase):
    """Tests related to :meth:`~coreax.refine.RefineReverse`."""

    @pytest.fixture
    def refine_method(self):
        return coreax.refine.RefineReverse()
