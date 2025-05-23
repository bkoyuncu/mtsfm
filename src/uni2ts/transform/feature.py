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
from einops import repeat

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, CollectFuncMixin


@dataclass
class AddVariateIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add variate_id to data_entry
    """

    fields: tuple[str, ...]
    max_dim: int
    optional_fields: tuple[str, ...] = tuple()
    variate_id_field: str = "variate_id"
    expected_ndim: int = 2
    randomize: bool = True
    collection_type: type = list
    #HERE we create variate indexes
    #adds 'variate_id' to data_entry, data_entry['variate_id']['target'] has shape (ch, #patches)
    #TODO we want to fix this always the same ones, so give randomize false
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.counter = 0

        self.dimensions = (
            np.random.choice(self.max_dim, size=self.max_dim, replace=False)
            if self.randomize
            else list(range(self.max_dim))
        )

        data_entry[self.variate_id_field] = self.collect_func(
            self._generate_variate_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_variate_id(
        self, data_entry: dict[str, Any], field: str
    ) -> np.ndarray:
        arr = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        dim, time = arr.shape[:2]
        if self.counter + dim > self.max_dim:
            raise ValueError(
                f"Variate ({self.counter + dim}) exceeds maximum variate {self.max_dim}. "
            )
        field_dim_id = repeat(
            np.asarray(self.dimensions[self.counter : self.counter + dim], dtype=int),
            "var -> var time",
            time=time,
        )
        self.counter += dim
        return field_dim_id


@dataclass
class AddTimeIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add time_id to data_entry
    """

    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    time_id_field: str = "time_id"
    expected_ndim: int = 2
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """
        add sequence_id
        """
        #HERE time_id added
        #adds 'time_id' to data_entry, data_entry['time_id']['target] has shape (ch, #patches) values are range 0 to #patches for both
        
        data_entry[self.time_id_field] = self.collect_func(
            self._generate_time_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_time_id(self, data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        var, time = arr.shape[:2]
        field_seq_id = np.arange(time)
        field_seq_id = repeat(field_seq_id, "time -> var time", var=var)
        return field_seq_id


@dataclass
class AddObservedMask(CollectFuncMixin, Transformation):
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    observed_mask_field: str = "observed_mask"
    collection_type: type = list
    #HERE creates the observed mask (True if observed) for non nan values
    #adds 'observed_mask' in data_entry
    #it has a dict with keys 'target', shape is (ch, T_crop)
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        observed_mask = self.collect_func(
            self._generate_observed_mask,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        data_entry[self.observed_mask_field] = observed_mask



        # observed_mask = self.collect_func(
        #     self._generate_jump_indices,
        #     data_entry,
        #     self.fields,
        #     optional_fields=self.optional_fields,
        # )
        return data_entry

    @staticmethod
    def _generate_observed_mask(data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr = data_entry[field]
        return ~np.isnan(arr)

    # @staticmethod
    # def _generate_jump_indices(data_entry: dict[str, Any], field: str) -> np.ndarray:
    #     arr = np.zeros_like(data_entry[field])

    #     all_column_names = data_entry['column_names'][data_entry['sampled_indices_perm']]
    #     for i in range(data_entry['target'].shape):
    #         freq_column = all_column_names[i].split['_'][0]
    #         if freq_column == 'D':
    #             arr[i] = 1
    #         elif freq_column == 'W':
    #             arr[i] = 1
    #         elif freq_column == 'M':
    #             arr[i] = 1
    #         elif freq_column == 'Q':
    #             arr[i] = 1
    #         elif freq_column == 'A':
    #             arr[i] = 1

    #     return arr
