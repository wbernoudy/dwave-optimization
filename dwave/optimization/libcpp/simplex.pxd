# Copyright 2025 D-Wave
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

from libcpp cimport bool
from libcpp.vector cimport vector

from dwave.optimization.libcpp cimport span


cdef extern from "../src/simplex.hpp" namespace "dwave::optimization" nogil:

    cdef cppclass SolveResult:
        enum SolveStatus:
            pass
        enum SolutionStatus:
            pass
        SolveResult()
        SolveStatus solve_status
        ssize_t num_iterations

        SolutionStatus solution_status()
        const vector[double]& solution()
        double objective()
        bool feasible()

    SolveResult linprog(span[const double] c, span[const double] b_lb,
                        span[const double] A_data, span[const double] b_ub,
                        span[const double] A_eq_data, span[const double] b_eq,
                        span[const double] lb, span[const double] ub) except +
