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

"""Module containing coreset solvers."""
from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx

from coreax.coresets import Coreset, Coresubset
from coreax.data import Data

_Data = TypeVar("_Data", bound=Data)
_Coreset = TypeVar("_Coreset", Coreset, Coresubset)
_SolverState = TypeVar("_SolverState")


class Solver(eqx.Module, Generic[_Data, _Coreset]):
    """
    Base class for coreset solvers.

    Solver is generic on the type of data required by the reduce method, and the type of
    coreset returned, providing a convenient means to distringuish between solvers that
    take (weighted) data/supervised data, and those which produce coresets/coresubsets.
    """

    @abstractmethod
    def reduce(
        self, coreset_size: int, dataset: _Data, solver_state: _SolverState
    ) -> tuple[_Coreset, _SolverState]:
        """
        Reduce a dataset to a coreset - solve the coreset problem.

        :param coreset_size: The desired size of the solved coreset
        :param dataset: The (potentially weighted and supervised) data to generate the
            coreset from
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: a solved coreset and relevant intermediate solver state information
        """


class CoresubsetSolver(Solver[_Data, Coresubset], Generic[_Data]):
    """
    Solver which returns a :class:`coreax.coreset.Coresubset`.

    A convenience class for the most common solver type in this package.
    """


class RefinementSolver(CoresubsetSolver[_Data], Generic[_Data]):
    """
    A :class:`coreax.solvers.CoresubsetSolver` which supports refinement.

    Some solvers assume implicitly/explicitly an initial coresubset on which the
    solution is dependant. Such solvers can be interpreted as refining the initial
    coresubset to produce another (solution) coresubset.

    By providing a 'refine' method, one can compose the results of different solvers
    together, and/or repeatedly apply/chain the result of a refinement based solve.
    """

    @abstractmethod
    def refine(
        self, coreset: Coreset | Coresubset, solver_state: _SolverState
    ) -> tuple[Coresubset, _SolverState]:
        """
        Refine a coreset - swap/update coresubset indices.

        :param coresubset: The coresubset to refine
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: a refined coresubset and relevant intermediate solver state information
        """
