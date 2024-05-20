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
Classes for reading different structures of input data.

In order to calculate a coreset, :meth:`~coreax.reduction.Coreset.fit` requires an
instance of a subclass of :class:`DataReader`. It is necessary to use
:class:`DataReader` because :class:`~coreax.reduction.Coreset` requires a
two-dimensional :class:`~jax.Array`. Data reductions are performed along the first
dimension.

The user should read in their data files using their preferred library that returns a
:class:`jax.Array` or :func:`numpy.array`. This array is passed to a
:meth:`load() <DataReader.load>` method. The user should not normally invoke
:class:`DataReader` directly. The user should select an appropriate subclass
of :class:`DataReader` to match the structure of the input array. The
:meth:`load() <DataReader.load>` method on the subclass will rearrange the original data
into the required two-dimensional format.

Various post-processing methods may be implemented if applicable to visualise or
restore a calculated coreset to match the format of the original data. To save a
copy of a coreset, call :meth:`format() <DataReader.format>` on a subclass to return an
:class:`~jax.Array`, which can be passed to the chosen IO library to write a file.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from typing import Any, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.typing import ArrayLike
from jaxtyping import Array, Shaped


class Data(eqx.Module):
    r"""
    Class for representing unsupervised data.

    A dataset of size `n` consists of a set of pairs :math:`\{(x_i, w_i)\}_{i=1}^n`
    where :math`x_i` are the features or inputs and :math:`w_i` are weights.

    :param data: An :math:`n \times d` array defining the features of the unsupervised
        dataset; d-vectors are converted to :math:`1 \times d` arrays
    :param weights: An :math:`n`-vector of weights where each element of the weights
        vector is paired with the corresponding index of the data array, forming the
        pair :math:`(x_i, w_i)`; if passed a scalar weight, it will be broadcast to an
        :math:`n`-vector. the default value of :data:`None` sets the weights to
        the ones vector (implies a scalar weight of one);
    """

    data: Shaped[Array, " n *d"]
    weights: Shaped[Array, " n"]

    def __init__(
        self,
        data: Shaped[ArrayLike, " n *d"],
        weights: Shaped[ArrayLike, " n"] | None = None,
    ):
        """Initialise Data class."""
        self.data = jnp.asarray(data)
        n = self.data.shape[:1]
        self.weights = jnp.broadcast_to(1 if weights is None else weights, n)

    def __getitem__(self, key: Any):
        """Array indexing behaviour."""
        return jtu.tree_map(lambda x: x[key], self)

    def __jax_array__(self) -> Shaped[ArrayLike, " n d"]:
        """Register ArrayLike behaviour - return value for `jnp.asarray(Data(...))`."""
        return self.data

    def __len__(self):
        """Return data length."""
        return len(self.data)

    def normalize(self) -> Data:
        """Return a copy of 'self' with 'weights' that sum to one."""
        normalized_weights = self.weights / jnp.sum(self.weights)
        return eqx.tree_at(lambda x: x.weights, self, normalized_weights)


_Data = TypeVar("_Data", bound=Data)


def as_data(x: Any, data_type: type[_Data] = Data) -> _Data:
    """Cast 'x' to a data instance."""
    return x if isinstance(x, data_type) else data_type(x)


def is_data(x: Any | Data):
    """Return 'True' if element is an instance of 'coreax.data.Data'."""
    return isinstance(x, Data)


class SupervisedData(Data):
    r"""
    Class for representing supervised data.

    A supervised dataset of size `n` consists of a set of triples
    :math:`\{(x_i, y_i, w_i)\}_{i=1}^n` where :math`x_i` are the features or inputs,
    :math:`y_i` are the responses or outputs, and :math:`w_i` are weights which
    correspond to the pairs :math:`(x_i, y_i)`.

    :param data: An :math:`n \times d` array defining the features of the supervised
        dataset paired with the corresponding index of the supervision;  d-vectors are
        converted to :math:`1 \times d` arrays
    :param supervision: An :math:`n \times p` array defining the responses of the
        supervised paired with the corresponding index of the data; d-vectors are
        converted to :math:`1 \times d` arrays
    :param weights: An :math:`n`-vector of weights where each element of the weights
        vector is is paired with the corresponding index of the data and supervision
        array, forming the triple :math:`(x_i, y_i, w_i)`; if passed a scalar weight,
        it will be broadcast to an :math:`n`-vector. the default value of :data:`None`
        sets the weights to the ones vector (implies a scalar weight of one);
    """

    supervision: Shaped[Array, " n *p"] = eqx.field(converter=jnp.atleast_2d)

    def __init__(
        self,
        data: Shaped[Array, " n d"],
        supervision: Shaped[Array, " n *p"],
        weights: Shaped[Array, " n"] | None = None,
    ):
        """Initialise SupervisedData class."""
        self.supervision = supervision
        super().__init__(data, weights)

    def __check_init__(self):
        """Check leading dimensions of supervision and data match."""
        if self.supervision.shape[0] != self.data.shape[0]:
            raise ValueError(
                "Leading dimensions of 'supervision' and 'data' must be equal"
            )
