from abc import ABCMeta
from typing import Any, TypeVar, Generic

import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from sklearn.neighbors import KDTree

from coreax.coreset import Coreset, Coresubset
from coreax.data import Data
from coreax.solvers.base import Solver

_Data = TypeVar("_Data", bound=Data)
_Coreset = TypeVar("_Coreset", Coreset, Coresubset)
_SolverState = TypeVar("_SolverState")


class CompositeSolver(Solver[_Data, _Coreset], Generic[_Data, _Coreset]):
    """Base class for solvers that compose/wrap other solvers."""

    base_solver: Solver[_Data, _Coreset]


class MapReduce(CompositeSolver[_Data, _Coreset], Generic[_Data, _Coreset]):
    r"""
    Calculate coreset of a given number of points using scalable reduction on blocks.

    This is a less memory-intensive alternative to :class:`SizeReduce`.

    It uses a :class:`~sklearn.neighbors.KDTree` to partition the original data into
    patches. Upon each of these a coreset of size :attr:`coreset_size` is calculated.
    These coresets are concatenated to produce a larger coreset covering the whole of
    the original data, which thus has size greater than :attr:`coreset_size`. This
    coreset is now treated as the original data and reduced recursively until its
    size is equal to :attr:`coreset_size`.

    :attr:`coreset_size` < :attr:`leaf_size` to ensure the algorithm converges.

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

    :param leaf_size: Approximate number of points to include in each partition;
        corresponds to ``leaf_size`` in :class:`~sklearn.neighbors.KDTree`;
        actual partition sizes vary non-strictly between :attr:`leaf_size` and
        :math:`2 \,\times` :attr:`leaf_size`; must be greater than :attr:`coreset_size`
    :param parallel: If :data:`True`, calculate coresets on partitions in parallel
    """

    base_solver: Solver[_Data, _Coreset]
    leaf_size: int
    devices: Array | None

    def reduce(
        self, coreset_size: int, dataset: _Data, solver_state: _SolverState
    ) -> tuple[_Coreset, _SolverState]:
        input_data = dataset.data
        partitioned_dataset = _jitable_kdtree(input_data, self.leaf_size)
        devices = jax.devices() if self.devices is None else self.devices
        mesh = jax.sharding.Mesh(np.asarray(devices).reshape(-1), axis_names=("i"))
        map_coreset_size = coreset_size // partitioned_dataset.shape[0]
        sharded_solver = shard_map(
            jtu.Partial(
                self.base_solver.reduce, map_coreset_size, solver_state=solver_state
            ),
            mesh=mesh,
            in_specs=jax.sharding.PartitionSpec("i"),
            out_specs=jax.sharding.PartitionSpec("i"),
        )
        return sharded_solver(partitioned_dataset)


def _jitable_kdtree(dataset, leaf_size):
    shape = (dataset.data.shape[0] // leaf_size, leaf_size)
    result_shape = jax.ShapeDtypeStruct(shape, jnp.int32)

    def _kdtree(_input_data):
        _, node_indices, _, _ = KDTree(_input_data, leaf_size=leaf_size).get_arrays()
        return node_indices.reshape(-1, leaf_size).astype(np.int32)

    indices = jax.pure_callback(_kdtree, result_shape, dataset.data)
    return jtu.tree_map(lambda x, indices: x[indices], dataset, indices)
