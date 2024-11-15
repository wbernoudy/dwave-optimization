# distutils: language = c++
# distutils: include_dirs = dwave/optimization/include/

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

cdef extern from "Python.h" nogil:
    ctypedef struct PyObject


cdef extern from "dwave-optimization/python_exception_handling.hpp" namespace "dwave::optimization::exception_handling":
    cdef PyObject* UnsupportedNaryReduceExpressionPyExc
    cdef void create_custom_exceptions()
    cdef void custom_exception_handler()
