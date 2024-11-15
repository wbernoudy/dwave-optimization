// Copyright 2023 D-Wave Systems Inc.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

#include <Python.h>

#include "dwave-optimization/nodes/lambda.hpp"

namespace dwave::optimization::exception_handling {

// Design inspired by the helpful example found here:
// https://gist.github.com/vyasr/4eddbff25bb197e056eab3ac1dda6f40

PyObject* UnsupportedNaryReduceExpressionPyExc;

void create_custom_exceptions() {
    UnsupportedNaryReduceExpressionPyExc = PyErr_NewException("dwave.optimization.libcpp.python_exception_handling.UnsupportedNaryReduceExpressionPyExc", NULL, NULL);
}

void custom_exception_handler() {
    // Taken from Cython source code 
    // https://github.com/cython/cython/blob/1c9dd6ab8b0613832c3033f678b354f23e3ce4c0/Cython/Utility/CppSupport.cpp#L10C1-L42C4
    //
    // Catch a handful of different errors here and turn them into the
    // equivalent Python errors.
    try {
        if (PyErr_Occurred())
            ; // let the latest Python exn pass through and ignore the current one
        else
            throw;
    } catch (const std::bad_alloc& exn) {
        PyErr_SetString(PyExc_MemoryError, exn.what());
    } catch (const std::bad_cast& exn) {
        PyErr_SetString(PyExc_TypeError, exn.what());
    } catch (const std::bad_typeid& exn) {
        PyErr_SetString(PyExc_TypeError, exn.what());
    } catch (const std::domain_error& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
    } catch (const std::invalid_argument& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
    } catch (const std::ios_base::failure& exn) {
        // Unfortunately, in standard C++ we have no way of distinguishing EOF
        // from other errors here; be careful with the exception mask
        PyErr_SetString(PyExc_IOError, exn.what());
    } catch (const std::out_of_range& exn) {
        // Change out_of_range to IndexError
        PyErr_SetString(PyExc_IndexError, exn.what());
    } catch (const std::overflow_error& exn) {
        PyErr_SetString(PyExc_OverflowError, exn.what());
    } catch (const std::range_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
    } catch (const std::underflow_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
    }

    // Now catch any custom exceptions
    catch (const dwave::optimization::UnsupportedNaryReduceExpressionError& exn) {
        PyErr_SetString(UnsupportedNaryReduceExpressionPyExc, exn.what());
    }

    // Finally catch general exceptions
    catch (const std::exception& exn) {
        PyErr_SetString(PyExc_RuntimeError, exn.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
    }
}

}  // namespace dwave::optimization::exception_handling
