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

"""Solvers for constructing coresubsets."""
from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from coreax.coreset import Coresubset
from coreax.data import Data
from coreax.kernel import Kernel
from coreax.solvers.base import RefinementSolver
from coreax.util import KeyArrayLike


class SolverState(NamedTuple):
    gramian_row_mean: Array | None = None


NULL = SolverState()


class KernelHerding(RefinementSolver[Data]):
    r"""
    Apply kernel herding to a dataset.

    Kernel herding is a deterministic, iterative and greedy approach to determine this
    compressed representation.

    Given one has selected :math:`T` data points for their compressed representation of
    the original dataset, kernel herding selects the next point as:

    .. math::

        x_{T+1} = \arg\max_{x} \left( \mathbb{E}[k(x, x')] -
            \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) \right)

    where :math:`k` is the kernel used, the expectation :math:`\mathbb{E}` is taken over
    the entire dataset, and the search is over the entire dataset. This can informally
    be seen as a balance between using points at which the underlying density is high
    (the first term) and exploration of distinct regions of the space (the second term).

    This class works with all children of :class:`~coreax.kernel.Kernel`, including
    Stein kernels.

    :param random_key: Key for random number generation
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param unique: Boolean that enforces the resulting coreset will only contain
        unique elements
    """

    random_key: KeyArrayLike
    kernel: Kernel
    unique: bool = True

    @eqx.filter_jit
    def reduce(
        self, coreset_size: int, dataset: Data, solver_state: SolverState = NULL
    ) -> tuple[Coresubset, SolverState]:
        r"""
        Execute kernel herding algorithm with Jax.

        We first compute the kernel matrix row sum mean if it is not given, and then
        iteratively add points to the coreset, balancing selecting points in high
        density regions with selecting points far from those already in the coreset.

        :param coreset_size: The size of the of coreset to generate
        """
        initial_coresubset = _initial_coresubset(coreset_size, dataset)
        return self.refine(initial_coresubset, solver_state)

    def refine(
        self, coresubset: Coresubset, solver_state: SolverState = NULL
    ) -> tuple[Coresubset, SolverState]:
        gramian_row_mean = solver_state.gramian_row_mean
        if gramian_row_mean is None:
            # TODO: Handling of block_size and unroll
            data = coresubset.pre_coreset_data.data
            gramian_row_mean = self.kernel.gramian_row_mean(data)

        def selection_function(i: int, _kernel_similiarity_penalty: ArrayLike) -> Array:
            return jnp.argmax(
                gramian_row_mean - _kernel_similiarity_penalty / (i + 1)
            )

        return _greedy_kernel_selection(
            coresubset,
            selection_function,
            self.kernel,
            gramian_row_mean,
            self.unique,
        )


class SteinThinning(RefinementSolver[Data]):
    r"""
    Apply regularised Stein thinning to a dataset.

    Stein thinning is a deterministic, iterative and greedy approach to determine this
    compressed representation.

    Given one has selected :math:`T` data points for their compressed representation of
    the original dataset, (regularised) Stein thinning selects the next point as:

    .. math::

        x_{T+1} = \arg\min_{x} \left( k_P(x, x) / 2 + \Delta^+ \log p(x) -
            \lambda T \log p(x) + \frac{1}{T+1}\sum_{t=1}^T k_P(x, x_t) \right)

    where :math:`k` is the Stein kernel induced by the supplied base kernel,
    :math:`\Delta^+` is the non-negative Laplace operator, :math:`\lambda` is a
    regularisation parameter, and the search is over the entire dataset.

    This class works with all children of :class:`~coreax.kernel.Kernel`, including
    Stein kernels.

    :param random_key: Key for random number generation
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param unique: Boolean that enforces the resulting coreset will only contain unique
        elements
    :param regularise: Boolean that enforces regularisation, as in
        :cite:`benard2023kernel`.
    """

    random_key: coreax.util.KeyArrayLike
    kernel: coreax.kernel.SteinKernel
    unique: bool = True
    regularise: bool = True

    def reduce(
        self, coreset_size: int, dataset: Data, solver_state
    ) -> tuple[Coresubset, SolverState]:
        initial_coresubset = _initial_coresubset(coreset_size, dataset)
        return self.refine(initial_coresubset, solver_state)

    def refine(
        self, coresubset: Coresubset, solver_state: SolverState = NULL
    ) -> tuple[Coresubset, SolverState]:
        r"""
        Execute Stein thinning algorithm with Jax.

        We first compute a score function, and then the Stein kernel. This is used to
        greedily choose points in the coreset to minimise kernel Stein discrepancy
        (KSD).

        :param coreset_size: The size of the of coreset to generate
        """
        data = coresubset.pre_coreset_data.data
        base_kernel = self.kernel.base_kernel
        score_function = self.kernel.score_function
        # We cannot guarantee the base kernel will have a length scale.
        # If it doesn't, we are safe to default to 'None'.
        bandwidth_method = getattr(base_kernel, "length_scale", None)

        @jax.vmap
        def _laplace_positive(x_):
            hessian = jax.jacfwd(score_function)(x_)
            return jnp.clip(jnp.diag(hessian), a_min=0.0).sum()

        def selection_function(i: int, _kernel_similarity_penalty: ArrayLike) -> Array:
            stein_kernel_diagonal = jax.vmap(self.kernel.compute_elementwise)(data, data)
            regularised_log_pdf = 0.0
            if self.regularise:
                stein_kernel_diagonal += _laplace_positive(data)
                # Fit a KDE to estimate log PDF: note the transpose
                kde = gaussian_kde(data.T, bw_method=bandwidth_method)
                # Use regularisation parameter suggested in :cite:`benard2023kernel`
                regulariser_lambda = 1 / len(coresubset)
                regularised_log_pdf = regulariser_lambda * kde.logpdf(data.T)
            kernel_stein_discrepancy = (
                stein_kernel_diagonal
                + 2.0 * _kernel_similarity_penalty
                - i * regularised_log_pdf
            )
            return kernel_stein_discrepancy

        return _greedy_kernel_selection(
            coresubset,
            selection_function,
            self.kernel,
            solver_state.gramian_row_mean,
            self.unique,
        )



def _initial_coresubset(coreset_size: int, dataset: Data) -> Coresubset:
    try:
        initial_coresubset_indices = jnp.zeros(coreset_size, dtype=jnp.int32)
    except TypeError as err:
        if coreset_size <= 0 or isinstance(coreset_size, float):
            raise ValueError("'coreset_size' must be a positive integer") from err
        raise
    return Coresubset(initial_coresubset_indices, dataset)


def _greedy_kernel_selection(
    coresubset: Coresubset,
    selection_function: Callable[[int, ArrayLike], Array],
    kernel: Kernel,
    gramian_row_mean: Array,
    unique: bool,
) -> tuple[Coresubset, SolverState]:
    data = coresubset.pre_coreset_data.data
    coreset_size = len(coresubset)
    kernel_similarity_penalty = jnp.zeros(len(data))

    def _greedy_body(i: int, val: tuple[Array, Array]) -> tuple[Array, ArrayLike]:
        coreset_indices, kernel_similarity_penalty = val
        updated_coreset_index = selection_function(i, kernel_similarity_penalty)
        updated_coreset_indices = coreset_indices.at[i].set(updated_coreset_index)
        updated_penalty = kernel_similarity_penalty + jnp.ravel(
            kernel.compute(data, data[updated_coreset_index])
        )
        if unique:
            # Prevent the same 'updated_coreset_index' from being selected on a
            # subsequent itteration, by setting the penalty to infinty.
            updated_penalty = updated_penalty.at[updated_coreset_index].set(jnp.inf)
        return updated_coreset_indices, updated_penalty

    initial_state = (coresubset.unweighted_indices, kernel_similarity_penalty)
    output_state = jax.lax.fori_loop(0, coreset_size, _greedy_body, initial_state)
    updated_coreset_indices, kernel_similarity_penalty = output_state
    updated_coreset = Coresubset(updated_coreset_indices, coresubset.pre_coreset_data)
    updated_solver_state = SolverState(gramian_row_mean)
    return updated_coreset, updated_solver_state


# # © Crown Copyright GCHQ
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# r"""
# Classes and associated functionality to construct coresubsets.

# Given a :math:`n \times d` dataset, one may wish to construct a compressed
# :math:`m \times d` representation of this dataset, where :math:`m << n`. This
# compressed representation is often referred to as a coreset. When the elements of a
# coreset are required to be elements of the original dataset, we denote this a
# coresubset. This module contains implementations of approaches to construct coresubsets.
# Coresets and coresubset are a type of data reduction, and these inherit from
# :class:`~coreax.reduction.Coreset`. The aim is to select a small set of indices
# that represent the key features of a larger dataset.

# The abstract base class is :class:`~coreax.reduction.Coreset`.
# """

# from coreax.data import Data
# from coreax.reduction import SolverState, RefinementSolver

# NULL = SolverState()

# class KernelHerding(RefinementSolver[Data]):
#     r"""
#     Apply kernel herding to a dataset.

#     Kernel herding is a deterministic, iterative and greedy approach to determine this
#     compressed representation.

#     Given one has selected :math:`T` data points for their compressed representation of
#     the original dataset, kernel herding selects the next point as:

#     .. math::

#         x_{T+1} = \arg\max_{x} \left( \mathbb{E}[k(x, x')] -
#             \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) \right)

#     where :math:`k` is the kernel used, the expectation :math:`\mathbb{E}` is taken over
#     the entire dataset, and the search is over the entire dataset. This can informally
#     be seen as a balance between using points at which the underlying density is high
#     (the first term) and exploration of distinct regions of the space (the second term).

#     This class works with all children of :class:`~coreax.kernel.Kernel`, including
#     Stein kernels.

#     :param random_key: Key for random number generation
#     :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
#         function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
#     :param block_size: Size of matrix blocks to process when computing the kernel
#         matrix row sum mean. Larger blocks will require more memory in the system.
#     :param unique: Boolean that enforces the resulting coreset will only contain
#         unique elements
#     """

#     random_key: coreax.util.KeyArrayLike
#     kernel: coreax.kernel.Kernel
#     block_size: int = 10_000
#     unique: bool = True

#     def reduce(
#         self, coreset_size: int, dataset: Data, solver_state: SolverState = NULL
#     ) -> tuple[Coresubset, SolverState]:
#         r"""
#         Execute kernel herding algorithm with Jax.

#         We first compute the kernel matrix row sum mean if it is not given, and then
#         iteratively add points to the coreset, balancing selecting points in high
#         density regions with selecting points far from those already in the coreset.

#         :param coreset_size: The size of the of coreset to generate
#         """
#         initial_coresubset = _initial_coresubset(coreset_size, dataset)
#         return self.refine(initial_coresubset, solver_state)

#     def refine(
#         self, coresubset: Coresubset, solver_state: SolverState = NULL
#     ) -> tuple[Coresubset, SolverState]:
#         kernel_matrix_row_sum_mean = solver_state.kernel_matrix_row_sum_mean
#         if kernel_matrix_row_sum_mean is None:
#             data = coresubset.pre_coreset_data.data
#             kmrsm = self.kernel.calculate_kernel_matrix_row_sum_mean
#             kernel_matrix_row_sum_mean = kmrsm(data, max_size=self.block_size)

#         def selection_function(i: int, _kernel_similiarity_penalty: ArrayLike) -> Array:
#             return jnp.argmax(
#                 kernel_matrix_row_sum_mean - _kernel_similiarity_penalty / (i + 1)
#             )

#         return _greedy_kernel_selection(
#             coresubset,
#             selection_function,
#             self.kernel,
#             kernel_matrix_row_sum_mean,
#             self.unique,
#         )


# class RandomSample(CoresubsetSolver[Data]):
#     r"""
#     Reduce a dataset by uniformly randomly sampling a fixed number of points.

#     .. note::
#         Any value other than :data:`True` will lead to random sampling with replacement
#         of points from the original data to construct the coreset.

#     :param random_key: Pseudo-random number generator key for sampling
#     :param unique: If :data:`True`, this flag enforces unique elements, i.e. sampling
#         without replacement
#     """

#     random_key: coreax.util.KeyArrayLike
#     unique: bool = True

#     def reduce(
#         self, coreset_size: int, dataset: Data, solver_state: SolverState = NULL
#     ) -> tuple[Coresubset, SolverState]:
#         """
#         Reduce a dataset by uniformly randomly sampling a fixed number of points.

#         This class is updated in-place. The randomly sampled points are stored in the
#         ``reduction_indices`` attribute.

#         :param coreset_size: The size of the of coreset to generate
#         """
#         num_data_points = len(dataset.data)
#         try:
#             coreset_indices = random.choice(
#                 self.random_key,
#                 a=jnp.arange(0, num_data_points),
#                 shape=(coreset_size,),
#                 replace=not self.unique,
#                 p=dataset.weights
#             )
#         # TypeError is raised if the size input to random.choice is negative, and an
#         # AttributeError is raised if the shape is a float.
#         except (AttributeError, TypeError) as exception:
#             if coreset_size <= 0 or not isinstance(coreset_size, int):
#                 raise ValueError(
#                     "coreset_size must be a positive integer"
#                 ) from exception
#             raise
#         return Coresubset(coreset_indices, dataset), solver_state


# class RPCholesky(RefinementSolver[Data]):
#     r"""
#     Apply Randomly Pivoted Cholesky (RPCholesky) to a dataset.

#     RPCholesky is a stochastic, iterative and greedy approach to determine this
#     compressed representation.

#     Given a dataset :math:`X` with :math:`N` data-points, and a desired coreset size of
#     :math:`M`, RPCholesky determines which points to select to constitute this coreset.

#     This is done by first computing the kernel Gram matrix of the original data, and
#     isolating the diagonal of this. A 'pivot point' is then sampled, where sampling
#     probabilities correspond to the size of the elements on this diagonal. The
#     data-point corresponding to this pivot point is added to the coreset, and the
#     diagonal of the Gram matrix is updated to add a repulsion term of sorts -
#     encouraging the coreset to select a range of distinct points in the original data.
#     The pivot sampling and diagonal updating steps are repeated until :math:`M` points
#     have been selected.

#     :param random_key: Key for random number generation
#     :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
#         function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
#     :param unique: Boolean that enforces the resulting coreset will only contain unique
#         elements
#     """

#     random_key: coreax.util.KeyArrayLike
#     kernel: coreax.kernel.Kernel
#     unique: bool = True

#     def reduce(
#         self, coreset_size: int, dataset: Data, solver_state: SolverState = NULL
#     ) -> tuple[Coresubset, SolverState]:
#         r"""
#         Execute RPCholesky algorithm with Jax.

#         Computes a low-rank approximation of the Gram matrix.

#         :param coreset_size: The size of the of coreset to generate
#         """
#         initial_coresubset = _initial_coresubset(coreset_size, dataset)
#         return self.refine(initial_coresubset, solver_state)

#     def refine(
#         self, coresubset: Coresubset, solver_state: SolverState = NULL
#     ) -> tuple[Coresubset, SolverState]:
#         coreset_indices = coresubset.indices.data
#         coreset_size = len(coreset_indices)
#         dataset = coresubset.pre_coreset_data
#         data = dataset.data
#         num_data_points = len(data)

#         def _greedy_body(
#             i: int, val: tuple[Array, Array, Array]
#         ) -> tuple[Array, Array, Array]:
#             r"""
#             Execute main loop of RPCholesky.

#             This function carries out one iteration of RPCholesky, defined in Algorithm
#             1 of :cite:`chen2023randomly`.

#             :param i: Loop counter
#             :param val: Tuple containing a :math:`m \times 1` array of the residual
#                 diagonal, :math:`n \times m` Cholesky matrix F, and a :math:`1 \times m`
#                 array of coreset indices. The ``i``-th element of the coreset_indices
#                 gets updated to the index of the selected coreset point in iteration
#                 ``i``.
#             :returns: Updated loop variables ``residual_diagonal``, ``F``,
#                 ``current_coreset_indices`` and ``key``
#             """
#             residual_diagonal, approximation_matrix, coreset_indices = val
#             key = random.fold_in(self.random_key, i)
#             pivot_point = random.choice(
#                 key, num_data_points, (), p=residual_diagonal, replace=False
#             )
#             updated_coreset_indices = coreset_indices.at[i].set(pivot_point)
#             # Remove overlap with previously chosen columns
#             g = (
#                 self.kernel.compute(data, data[pivot_point])
#                 - (approximation_matrix @ approximation_matrix[pivot_point])[:, None]
#             )
#             updated_approximation_matrix = approximation_matrix.at[:, i].set(
#                 jnp.ravel(g / jnp.sqrt(g[pivot_point]))
#             )
#             # Track diagonal of residual matrix and ensure it remains non-negative
#             updated_residual_diagonal = jnp.clip(
#                 residual_diagonal - jnp.square(approximation_matrix[:, i]), a_min=0
#             )
#             if self.unique:
#                 # ensures that index selected_pivot_point can't be drawn again in future
#                 residual_diagonal = residual_diagonal.at[pivot_point].set(0.0)
#             return (
#                 updated_residual_diagonal,
#                 updated_approximation_matrix,
#                 updated_coreset_indices,
#             )

#         approximation_matrix = jnp.zeros((num_data_points, coreset_size))
#         # This could be given as solver state?
#         residual_diagonal = vmap(self.kernel.compute_elementwise)(data, data)
#         initial_state = (residual_diagonal, approximation_matrix, coreset_indices)
#         output_state = lax.fori_loop(0, coreset_size, _greedy_body, initial_state)
#         _, _, updated_coreset_indices = output_state
#         updated_coreset = Coresubset(updated_coreset_indices, dataset)
#         return updated_coreset, solver_state