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

r"""
Module provides tools for reducing a large dataset down to a coreset.

To prepare data for reduction, convert it into a :class:`~jax.Array` and pass to an
appropriate instance of :class:`~coreax.data.DataReader`. The class will convert the
data internally to be an :math:`n \times d` :class:`~jax.Array`. The resulting coreset
will be :math:`m \times d` where :math:`m \ll n` but still retain similar statistical
properties.

The user selects a method by choosing a :class:`Coreset` and a
:class:`ReductionStrategy`. For example, the user may obtain a uniform random
sample of :math:`m` points by using the :class:`SizeReduce` strategy and a
:class:`~coreax.coresubset.RandomSample` coreset. This may be implemented for
:class:`~coreax.data.ArrayData` by calling

.. code-block:: python

    original_data = ArrayData.load(input_data)
    coreset = RandomSample()
    coreset.reduce(original_data, SizeReduce(m))
    print(coreset.format())

:class:`ReductionStrategy` and :class:`Coreset` are abstract base classes defining the
interface for which particular methods can be implemented.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from multiprocessing.pool import ThreadPool
from typing import TypeVar

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from sklearn.neighbors import KDTree
from typing_extensions import Self

import coreax.data
import coreax.kernel
import coreax.metrics
import coreax.util
import coreax.weights


class Coreset(ABC):
    r"""
    Abstract base class for coresets.

    TLDR: a coreset is a reduced set of :math:`\hat{n}` (potentially weighted) data
    points that, in some sense, best represent the "important" properties of a larger
    set of :math:`n > \hat{n}` (potentially weighted) data points.

    Given a dataset :math:`X = \{x_i\}_{i=1}^n, x \in \Omega`, where each node is paired
    with a non-negative (probability) weight :math:`w_i \in \mathbb{R} \ge 0`, there
    exists an implied discrete (probability) measure over :math:`\Omega`

    .. math:
        \eta_n = \sum_{i=1}^{n} w_i \delta_{x_i}.

    If we then specify a set of test-functions :math:`\Phi = {\phi_1, \dots, \phi_M}`,
    where :math:`\phi_i \colon \Omega \to \mathbb{R}`, which somehow capture the
    "important" properties of the data, then there also exists an implied push-forward
    measure over :math:`\mathbb{R}^M`

    .. math:
        \mu_n = \sum_{i=1}^{n} w_i \delta_{\Phi(x_i)}.

    A coreset is simply a reduced measure containing :math:`\hat{n} < n` updated nodes
    :math:`\hat{x}_i` and weights :math:`\hat{w}_i`, such that the push-forward measure
    of the coreset :math:`\nu_\hat{n}` has (approximately for some algorithms) the same
    "centre-of-mass" as the push-forward measure for the original data :math:`\mu_n`

    .. math:
        \text{CoM}(\mu_n) = \text{CoM}(\nu_\hat{n}),
        \text{CoM}(\nu_\hat{n}) = \sum_{i=1}^\hat{n} \hat{w}_i \delta_{\Phi(\hat{x}_i)}.

    Note: depending on the algorithm, the test-functions may be explicitly specified by
    the user, or implicitly defined by the algorithm's specific objectives.

    :param weights_optimiser: :class:`~coreax.weights.WeightsOptimiser` object to
        determine weights for coreset points to optimise some quality metric, or
        :data:`None` (default) if unweighted
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`, or
        :data:`None` if not applicable
    :param refine_method: :class:`~coreax.refine.Refine` object to use, or :data:`None`
        (default) if no refinement is required
    """

    def __init__(
        self,
        *,
        weights_optimiser: coreax.weights.WeightsOptimiser | None = None,
        kernel: coreax.kernel.Kernel | None = None,
        refine_method: "coreax.refine.Refine | None" = None,
    ):
        """Initialise class and set internal attributes to defaults."""
        self.weights_optimiser = weights_optimiser
        """
        Weights optimiser
        """
        self.kernel = kernel
        """
        Kernel
        """
        self.refine_method = refine_method
        """
        Refine method
        """

        # Data attributes not set in init
        self.original_data: coreax.data.DataReader | None = None
        """
        Data to be reduced
        """
        self.coreset: Array | None = None
        """
        Calculated coreset. The order of rows need not be monotonic with those in the
        original data (applicable only to coresubset).
        """

    def clone_empty(self) -> Self:
        """
        Create an empty copy of this class with all data removed.

        Other parameters are retained.

        .. warning:: This copy is shallow so :attr:`weights_optimiser` etc. still point
            to the original object.

        .. warning:: If any additional data attributes are added in a subclass, it
            should reimplement this method.

        :return: Copy of this class with data removed
        """
        new_obj = copy(self)
        new_obj.original_data = None
        new_obj.coreset = None
        return new_obj

    def fit(
        self, original_data: coreax.data.DataReader, strategy: ReductionStrategy
    ) -> None:
        """
        Compute coreset using a given reduction strategy.

        The resulting coreset is saved in-place to :attr:`coreset`.

        :param original_data: Instance of :class:`~coreax.data.DataReader` containing
            the data we wish to reduce
        :param strategy: Reduction strategy to use
        """
        self.original_data = original_data
        strategy.reduce(self)

    @abstractmethod
    def fit_to_size(self, coreset_size: int) -> None:
        """
        Compute coreset for a fixed target size.

        .. note:: The user should not normally call this method directly; call
            :meth:`fit` instead.

        This method is equivalent to calling :meth:`fit` with strategy
        :class:`SizeReduce` with ``coreset_size`` except that it requires
        :attr:`original_data` to already be populated.

        The resulting coreset is saved in-place to :attr:`coreset`.

        If ``coreset_size`` is greater than the number of data points in
        :attr:`original_data`, the resulting coreset may be larger than the original
        data, if the coreset method permits. A :exc:`ValueError` is raised if it is not
        possible to generate a coreset of size ``coreset_size``.

        If ``coreset_size`` is equal to the number of data points in
        :attr:`original_data`, the resulting coreset is not necessarily equal to the
        original data, depending on the coreset method, metric and weighting.

        :param coreset_size: Number of points to include in coreset
        :raises ValueError: When it is not possible to generate a coreset of size
            ``coreset_size``
        """

    def solve_weights(self) -> Array:
        """
        Solve for optimal weighting of points in :attr:`coreset`.

        :return: Optimal weighting of points in :attr:`coreset` to represent the
            original data
        """
        self.validate_fitted(Coreset.solve_weights.__name__)
        return self.weights_optimiser.solve(
            self.original_data.pre_coreset_array, self.coreset
        )

    def compute_metric(
        self,
        metric: coreax.metrics.Metric,
        block_size: int | None = None,
        weights_x: ArrayLike | None = None,
        weights_y: ArrayLike | None = None,
    ) -> Array:
        r"""
        Compute metric comparing the coreset with the original data.

        The metric is computed unweighted unless ``weights_x`` and/or ``weights_y`` is
        supplied as an array. Further options are available by calling the chosen
        :class:`~coreax.metrics.Metric` class directly.

        :param metric: Instance of :class:`~coreax.metrics.Metric` to use
        :param block_size: Size of matrix block to process, or :data:`None` to not split
            into blocks
        :param weights_x: An :math:`1 \times n` array of weights for associated points
            in ``x``, or :data:`None` if not required
        :param weights_y: An :math:`1 \times m` array of weights for associated points
            in ``y``, or :data:`None` if not required
        :return: Metric computed as a zero-dimensional array
        """
        self.validate_fitted(Coreset.compute_metric.__name__)
        return metric.compute(
            self.original_data.pre_coreset_array,
            self.coreset,
            block_size=block_size,
            weights_x=weights_x,
            weights_y=weights_y,
        )

    def format(self) -> Array:
        """
        Format coreset to match the shape of the original data.

        :return: Array of formatted data
        """
        self.validate_fitted(Coreset.format.__name__)
        return self.original_data.format(self)

    def render(self) -> None:
        """Plot coreset interactively using :mod:`matplotlib.pyplot`."""
        self.validate_fitted(Coreset.render.__name__)
        return self.original_data.render(self)

    def copy_fit(self, other: Self, deep: bool = False) -> None:
        """
        Copy fitted coreset from other instance to this instance.

        The other coreset must be of the same type as this instance and
        :attr:`original_data` must also be populated on ``other``. The user must ensure
        :attr:`original_data` is correctly populated on this instance.

        :param other: :class:`Coreset` from which to copy calculated coreset
        :param deep: If :data:`True`, make a shallow copy of :attr:`coreset`; otherwise,
            reference the same objects
        :raises TypeError: If ``other`` does not have the **exact same type**.
        """
        other.validate_fitted(Coreset.copy_fit.__name__ + " from another Coreset")
        if deep:
            self.coreset = copy(other.coreset)
        else:
            self.coreset = other.coreset

    def validate_fitted(self, caller_name: str) -> None:
        """
        Raise :exc:`~coreax.util.NotCalculatedError` if coreset has not been fitted yet.

        :param caller_name: Name of calling method to display in error message
        :raises NotCalculatedError: If :attr:`original_data` or :attr:`coreset` is
            :data:`None`
        """
        if not isinstance(self.original_data, coreax.data.DataReader) or not isinstance(
            self.coreset, Array
        ):
            raise coreax.util.NotCalculatedError(
                "Need to call "
                + Coreset.fit.__name__
                + f" before calling {caller_name}"
            )


class Coresubset(Coreset):
    r"""
    Abstract base class for coresubsets.

    A coresubset is a coreset, with the additional condition that the support of the
    reduced measure (the coreset), must be a subset of the support of the original
    measure (the original data), such that

    .. math:
        \hat{x}_i = x_i, \forall i \in I,
        I \subset \{1, \dots, n\}, text{card}(I) = \hat{n}.

    Thus, a coresubset, unlike a corset, ensures that feasibility constraints on the
    support of the measure are maintained :cite:`litterer2012recombination`. This is
    vital if, for example, the test-functions are only defined on the support of the
    original measure/nodes, rather than all of :math:`\Omega`.

    In coresubsets, the measure reduction can be implicit (setting weights/nodes to
    zero for all :math:`i \in I \ {1, \dots, n}`) or explicit (removing entries from the
    weight/node arrays). The implicit approach is useful when input/output array shape
    stability is required (E.G. for some JAX transformations); the explicit approach is
    more similar to a standard coreset.

    :param weights_optimiser: :class:`~coreax.weights.WeightsOptimiser` object to
        determine weights for coreset points to optimise some quality metric, or
        :data:`None` (default) if unweighted
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`, or
        :data:`None` if not applicable
    """

    def __init__(
        self,
        *,
        weights_optimiser: coreax.weights.WeightsOptimiser | None = None,
        kernel: coreax.kernel.Kernel | None = None,
        refine_method: "coreax.refine.Refine | None" = None,
    ):
        """Initialise class and set internal attributes to defaults."""
        self.coreset_indices: Array | None = None
        """
        Indices of :attr:`~Coreset.coreset` points in :attr:`~Coreset.original_data`, if
        applicable. The order matches the rows of :attr:`~Coreset.coreset`.
        """
        super().__init__(
            weights_optimiser=weights_optimiser,
            kernel=kernel,
            refine_method=refine_method,
        )

    def clone_empty(self) -> Self:
        """
        Create an empty copy of this class with all data removed.

        Other parameters are retained.

        .. warning:: This copy is shallow so :attr:`weights_optimiser` etc. still point
            to the original object.

        .. warning:: If any additional data attributes are added in a subclass, it
            should reimplement this method.

        :return: Copy of this class with data removed
        """
        new_obj = super().clone_empty()
        new_obj.coreset_indices = None
        return new_obj

    def copy_fit(self, other: Self, deep: bool = False) -> None:
        """
        Copy fitted coreset from other instance to this instance.

        The other coreset must be of the same type as this instance and
        :attr:`~Coreset.original_data` must also be populated on ``other``. The user
        must ensure :attr:`~Coreset.original_data` is correctly populated on this
        instance.

        :param other: :class:`Coreset` from which to copy calculated coreset
        :param deep: If :data:`True`, make a shallow copy of :attr:`~Coreset.coreset`
            and :attr:`coreset_indices`; otherwise, reference same objects
        :raises TypeError: If ``other`` does not have the **exact same type**.
        """
        super().copy_fit(other, deep=deep)
        if deep:
            self.coreset_indices = copy(other.coreset_indices)
        else:
            self.coreset_indices = other.coreset_indices

    def refine(self) -> None:
        """
        Refine coresubset.

        Only applicable to coreset methods that generate coresubsets.

        :attr:`~Coreset.coreset` is updated in place.

        :raises TypeError: When :attr:`~Coreset.refine_method` is :data:`None`
        """
        if self.refine_method is None:
            raise TypeError("Cannot refine without a refine_method")
        # Validate appropriate attributes are set on coreset inside refine_method.refine
        self.refine_method.refine(self)


C = TypeVar("C", bound=Coreset)


class ReductionStrategy(ABC):
    """
    Define a strategy for how to construct a coreset for a given type of coreset.

    The strategy determines the size of the coreset, approximation strategies to aid
    memory management and other similar aspects that wrap around the type of coreset.
    """

    @abstractmethod
    def reduce(self, coreset: Coreset) -> None:
        """
        Reduce a dataset to a coreset using this strategy.

        ``coreset`` is updated in place.

        :param coreset: :class:`Coreset` instance to populate in place
        """


class SizeReduce(ReductionStrategy):
    """
    Calculate coreset containing a given number of points.

    :param coreset_size: Number of points to include in coreset
    """

    def __init__(self, coreset_size: int):
        """Initialise class."""
        super().__init__()
        self.coreset_size = coreset_size

    def reduce(self, coreset: Coreset) -> None:
        """
        Reduce a dataset to a coreset using this strategy.

        ``coreset`` is updated in place.

        :param coreset: :class:`Coreset` instance to populate in place
        """
        coreset.fit_to_size(self.coreset_size)


class MapReduce(ReductionStrategy):
    r"""
    Calculate coreset of a given number of points using scalable reduction on blocks.

    This is a less memory-intensive alternative to :class:`SizeReduce`.

    It uses a :class:`~sklearn.neighbors.KDTree` to partition the original data into
    patches. Upon each of these a coreset of size :attr:`coreset_size` is calculated.
    These coresets are concatenated to produce a larger coreset covering the whole of
    the original data, which thus has size greater than :attr:`coreset_size`. This
    coreset is now treated as the original data and reduced recursively until its
    size is equal to :attr:`coreset_size`.

    :attr:`coreset_size` < :attr:`leaf_size` to ensure the algorithm converges. If
    for whatever reason you wish to break this restriction, use :class:`SizeReduce`
    instead.

    There is some intricate set-up:

    #.  :attr:`coreset_size` must be less than :attr:`leaf_size`.
    #.  Unweighted coresets are calculated on each patch of roughly
        :attr:`leaf_size` points and then concatenated. More specifically, each
        patch contains between :attr:`leaf_size` and
        :math:`2 \,\,\times` :attr:`leaf_size` points, inclusive.
    #.  Recursively calculate ever smaller coresets until a global coreset with size
        :attr:`coreset_size` is obtained.
    #.  If the input data on the final iteration is smaller than :attr:`coreset_size`,
        the whole input data is returned as the coreset and thus is smaller than the
        requested size.

    Let :math:`n_k` be the number of points after each recursion with :math:`n_0` equal
    to the size of the original data. Then, each recursion reduces the size of the
    coreset such that

    .. math::

        n_k <= \frac{n_{k - 1}}{\texttt{leaf_size}} \texttt{coreset_size},

    so

    .. math::

        n_k <= \left( \frac{\texttt{coreset_size}}{\texttt{leaf_size}} \right)^k n_0.

    Thus, the number of iterations required is roughly (find :math:`k` when
    :math:`n_k =` :attr:`coreset_size`)

    .. math::

        \frac{
            \log{\texttt{coreset_size}} - \log{\left(\text{original data size}\right)}
        }{
            \log{\texttt{coreset_size}} - \log{\texttt{leaf_size}}
        } .

    :param coreset_size: Number of points to include in coreset
    :param leaf_size: Approximate number of points to include in each partition;
        corresponds to ``leaf_size`` in :class:`~sklearn.neighbors.KDTree`;
        actual partition sizes vary non-strictly between :attr:`leaf_size` and
        :math:`2 \,\times` :attr:`leaf_size`; must be greater than :attr:`coreset_size`
    :param parallel: If :data:`True`, calculate coresets on partitions in parallel
    """

    def __init__(
        self,
        coreset_size: int,
        leaf_size: int,
        parallel: bool = True,
    ):
        """Initialise class."""
        super().__init__()
        self.coreset_size = coreset_size
        """
        Coreset size
        """
        self.leaf_size = leaf_size
        """
        Leaf size
        """
        self.parallel = parallel

    def reduce(self, coreset: Coreset) -> None:
        """
        Reduce a dataset to a coreset using scalable reduction.

        It is performed using recursive calls to :meth:`_reduce_recursive`.

        :param coreset: :class:`Coreset` instance to populate in place
        """
        input_data = coreset.original_data.pre_coreset_array
        # _reduce_recursive returns a copy of coreset so need to transfer calculated
        # coreset fit into the original coreset object
        coreset.copy_fit(
            self._reduce_recursive(
                template=coreset,
                input_data=input_data,
                input_indices=jnp.array(range(input_data.shape[0])),
            )
        )

    def _reduce_recursive(
        self,
        template: C,
        input_data: ArrayLike,
        input_indices: ArrayLike | None = None,
    ) -> C:
        r"""
        Recursively execute scalable reduction.

        :param template: Instance of :class:`Coreset` to duplicate
        :param input_data: Data to reduce on this iteration
        :param input_indices: Indices of ``input_data``, if applicable to ``template``
        :return: Copy of ``template`` containing fitted coreset
        """
        # Check if no partitions are required
        if input_data.shape[0] <= self.leaf_size:
            # Length of input_data < coreset_size is only possible if input_data is the
            # original data, so it is safe to request a coreset of size larger than the
            # original data (if of limited use)
            return self._coreset_copy_fit(template, input_data, input_indices)

        # Partitions required

        # Build a kdtree
        try:
            # Note that a TypeError is raised if the leaf_size input to KDTree is
            # negative
            kdtree = KDTree(input_data, leaf_size=self.leaf_size)
        except TypeError as exception:
            if isinstance(self.leaf_size, float):
                raise ValueError("leaf_size must be a positive integer") from exception
            raise
        _, node_indices, nodes, _ = kdtree.get_arrays()
        new_indices = [jnp.array(node_indices[nd[0] : nd[1]]) for nd in nodes if nd[2]]
        split_data = [input_data[n] for n in new_indices]

        # Generate a coreset on each partition
        if self.parallel:
            with ThreadPool() as pool:
                res = pool.map_async(
                    lambda args: self._coreset_copy_fit(template, *args),
                    zip(split_data, new_indices),
                )
                res.wait()
                partition_coresets: list[C] = res.get()
        else:
            partition_coresets = [
                self._coreset_copy_fit(template, sd, sd_indices)
                for sd, sd_indices in zip(split_data, new_indices)
            ]

        # Concatenate coresets
        full_coreset = jnp.concatenate([pc.coreset for pc in partition_coresets])
        if partition_coresets[0].coreset_indices is None:
            full_indices = None
        else:
            full_indices = jnp.concatenate(
                [input_indices[pc.coreset_indices] for pc in partition_coresets]
            )

        # Recursively reduce large coreset
        # coreset_indices will be None if not applicable to the coreset method
        return self._reduce_recursive(
            template=template, input_data=full_coreset, input_indices=full_indices
        )

    def _coreset_copy_fit(
        self, template: C, input_data: ArrayLike, input_indices: ArrayLike | None
    ) -> C:
        """
        Create a new instance of a :class:`Coreset` and fit to given data.

        If applicable to the coreset method, the coreset indices are overwritten using
        ``input_indices`` as the mapping to allow ``input_data`` to be a subset of the
        original data.

        :param template: Instance of :class:`Coreset` to duplicate
        :param input_data: Data to fit
        :param input_indices: Indices of ``input_data``, if applicable to ``template``
        :return: New instance of the coreset fitted to ``input_data``
        """
        coreset = template.clone_empty()
        coreset.original_data = coreax.data.ArrayData.load(input_data)
        coreset.fit_to_size(self.coreset_size)
        # Update indices
        if coreset.coreset_indices is not None:
            # Should not reach here if input_indices is not populated
            assert input_indices is not None
            coreset.coreset_indices = input_indices[coreset.coreset_indices]
        return coreset
