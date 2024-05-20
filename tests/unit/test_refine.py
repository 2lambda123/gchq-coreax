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
