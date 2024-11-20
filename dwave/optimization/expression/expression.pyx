# Copyright 2024 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numbers
from typing import Optional

from libcpp cimport bool
from libcpp.cast cimport dynamic_cast

from dwave.optimization.libcpp.array cimport Array as cppArray
from dwave.optimization.libcpp.graph cimport Node as cppNode
from dwave.optimization.libcpp.nodes cimport InputNode as cppInputNode
from dwave.optimization.model cimport ArraySymbol, _Model, States
from dwave.optimization.symbols cimport symbol_from_ptr

ctypedef cppNode* cppNodePtr


__all__ = ["Expression"]


cdef class Expression(_Model):
    def __init__(
        self,
        num_inputs: int = 0,
        # necessary to prevent Cython from rejecting an int
        lower_bound: Optional[numbers.Real] = None,
        upper_bound: Optional[numbers.Real] = None,
        integral: Optional[bool] = None,
    ):
        self.states = States(self)

        self._data_sources = []

        if num_inputs > 0:
            if any(arg is None for arg in (lower_bound, upper_bound, integral)):
                raise ValueError(
                    "`lower_bound`, `upper_bound` and `integral` must be provided "
                    "explicitly when initializing inputs"
                )
            for _ in range(num_inputs):
                self.input(lower_bound, upper_bound, integral)

    def input(self, lower_bound: float, upper_bound: float, bool integral):
        """TODO"""
        # avoid circular import
        from dwave.optimization.symbols import Input
        # Shape is always scalar for now
        return Input(self, lower_bound, upper_bound, integral, shape=tuple())

    def set_output(self, value: ArraySymbol):
        self.output = value

    cpdef Py_ssize_t num_inputs(self) noexcept:
        return self._graph.num_inputs()

    def iter_inputs(self):
        inputs = self._graph.inputs()
        for i in range(self._graph.num_inputs()):
            yield symbol_from_ptr(self, inputs[i])
