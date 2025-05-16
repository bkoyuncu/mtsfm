#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Num

from ._base import Transformation
from ._mixin import ApplyFuncMixin


class ImputationMethod:
    def __call__(
        self, x: Num[np.ndarray, "length *dim"]
    ) -> Num[np.ndarray, "length *dim"]: ...


@dataclass(frozen=True)
class DummyValueImputation(ImputationMethod):
    value: int | float | complex = 0.0

    def __call__(
        self, x: Num[np.ndarray, "length *dim"]
    ) -> Num[np.ndarray, "length *dim"]:
        x[np.isnan(x)] = self.value
        return x


@dataclass(frozen=True)
class LastValueImputation(ImputationMethod):
    value: int | float | complex = 0.0

    def __call__(
        self, x: Num[np.ndarray, "length *dim"]
    ) -> Num[np.ndarray, "length *dim"]:
        x = x.T
        x[0:1][np.isnan(x[0:1])] = self.value
        mask = np.isnan(x)
        idx = np.arange(len(x))
        if x.ndim == 2:
            idx = np.expand_dims(idx, axis=1)
        idx = np.where(~mask, idx, 0)
        idx = np.maximum.accumulate(idx, axis=0)
        if x.ndim == 2:
            x = x[idx, np.arange(x.shape[1])]
        else:
            x = x[idx]
        return x.T

@dataclass(frozen=True)
class CausalMeanImputation(ImputationMethod):
    # TODO: implement causal mean imputation
    value: int | float | complex = 0.0

    def __call__(
        self, x: Num[np.ndarray, "length *dim"]
    ) -> Num[np.ndarray, "length *dim"]:# ...
        x = x.T
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if np.isnan(x[i, j]):
                    if i == 0:
                        x[i, j] = self.value
                    else:
                        x[i, j] = np.nanmean(x[:i, j], axis=0)
        return x.T



# @dataclass(frozen=True)
# class CausalMeanImputation(ImputationMethod):
#     def __call__(
#         self, x: Num[np.ndarray, "length *dim"], value: Union[int, float, complex] = 0.0
#     ) -> Num[np.ndarray, "length *dim"]:
#         x = x.T
#         for i in range(x.shape[0]):
#             for j in range(x.shape[1]):
#                 if np.isnan(x[i, j]):
#                     if i == 0:
#                         x[i, j] = value
#                     else:
#                         x[i, j] = np.nanmean(x[:i, j], axis=0)
#         return x.T


@dataclass
class ImputeTimeSeries(ApplyFuncMixin, Transformation):
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    imputation_method: ImputationMethod = DummyValueImputation(value=0.0)
    #HERE nans are imputed with zeros, in_place with 'targets'
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.apply_func(
            self._impute,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _impute(self, data_entry: dict[str, Any], field: str):
        value = data_entry[field]
        nan_entries = np.isnan(value)
        if nan_entries.any():
            data_entry[field] = self.imputation_method(value)
