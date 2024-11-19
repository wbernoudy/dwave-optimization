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

from typing import Optional

from libcpp cimport bool

from dwave.optimization.libcpp.array cimport Array as cppArray
from dwave.optimization.symbols cimport symbol_from_ptr


__all__ = ["Expression"]


cdef class Expression(_Model):
    def __init__(self):
        pass

    def input(self, lower_bound: float, upper_bound: float, bool integral, shape: Optional[tuple] = None):
        """TODO"""
        # avoid circular import
        from dwave.optimization.symbols import Input
        return Input(self, lower_bound, upper_bound, integral, shape=shape)

    def set_output(self, value: ArraySymbol):
        self.output = value
