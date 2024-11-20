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

from dwave.optimization.model import _Model


_ShapeLike: typing.TypeAlias = typing.Union[int, collections.abc.Sequence[int]]


class Expression(_Model):
    def __init__(
        self,
        num_inputs: int = 0,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        integral: Optional[bool] = None,
    ): ...

    def input(self, lower_bound: float, upper_bound: float, integral: bool, shape: Optional[tuple] = None):

    @property
    def output(self) -> ArraySymbol: ...

    def set_output(self, value: ArraySymbol): ...
