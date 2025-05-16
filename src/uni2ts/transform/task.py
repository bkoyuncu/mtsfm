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
from functools import partial
from typing import Any

import numpy as np
from jaxtyping import Bool, Float

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin


@dataclass
class MaskedPrediction(MapFuncMixin, CheckArrNDimMixin, Transformation):
    min_mask_ratio: float
    max_mask_ratio: float
    target_field: str = "target"
    truncate_fields: tuple[str, ...] = tuple()
    optional_truncate_fields: tuple[str, ...] = tuple()
    prediction_mask_field: str = "prediction_mask"
    expected_ndim: int = 2
    #HERE adds prediction_mask to data_entry, has shape (ch, #patches) NO 'target' key
    #CRITICAL mask is temporal, aka it is on last L patches

    def __post_init__(self):
        assert (
            self.min_mask_ratio <= self.max_mask_ratio
        ), "min_mask_ratio must be <= max_mask_ratio"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target = data_entry[self.target_field]
        prediction_mask = self._generate_prediction_mask(target)
        self.map_func(
            partial(self._truncate, mask=prediction_mask),  # noqa
            data_entry,
            self.truncate_fields,
            optional_fields=self.optional_truncate_fields,
        )
        data_entry[self.prediction_mask_field] = prediction_mask
        return data_entry

    def _generate_prediction_mask(
        self, target: Float[np.ndarray, "var time *feat"]
    ) -> Bool[np.ndarray, "var time"]:
        self.check_ndim("target", target, self.expected_ndim)
        var, time = target.shape[:2]
        prediction_mask = np.zeros((var, time), dtype=bool)
        mask_ratio = np.random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        mask_length = max(1, round(time * mask_ratio)) 
        prediction_mask[:, -mask_length:] = True
        return prediction_mask

    def _truncate(
        self,
        data_entry: dict[str, Any],
        field: str,
        mask: np.ndarray,
    ) -> np.ndarray | list[np.ndarray] | dict[str, np.ndarray]:
        arr: np.ndarray | list[np.ndarray] | dict[str, np.ndarray] = data_entry[field]
        if isinstance(arr, list):
            return [self._truncate_arr(a, mask) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.truncate_fields or k in self.optional_truncate_fields:
                    arr[k] = self._truncate_arr(v, mask)
            return arr
        return self._truncate_arr(arr, mask)

    @staticmethod
    def _truncate_arr(
        arr: Float[np.ndarray, "var time *feat"], mask: Bool[np.ndarray, "var time"]
    ) -> Float[np.ndarray, "var time-mask_len *feat"]:
        return arr[:, ~mask[0]]


@dataclass
class ExtendMask(CheckArrNDimMixin, CollectFuncMixin, Transformation):
    fields: tuple[str, ...]
    mask_field: str
    optional_fields: tuple[str, ...] = tuple()
    expected_ndim: int = 2
    #HERE creates aux target mask, return empty for us, bc fields are empty #for me it just put it in a list?
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target_mask: np.ndarray = data_entry[self.mask_field] #shape (FEATURES, #patches)
        aux_target_mask: list[np.ndarray] = self.collect_func_list(
            self._generate_target_mask,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        data_entry[self.mask_field] = [target_mask] + aux_target_mask
        return data_entry

    def _generate_target_mask(
        self, data_entry: dict[str, Any], field: str
    ) -> np.ndarray:
        arr: np.ndarray = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        var, time = arr.shape[:2]
        field_target_mask = np.zeros((var, time), dtype=bool)
        return field_target_mask


@dataclass
class EvalMaskedPrediction(MapFuncMixin, CheckArrNDimMixin, Transformation):
    mask_length: int
    target_field: str = "target"
    truncate_fields: tuple[str, ...] = tuple()
    optional_truncate_fields: tuple[str, ...] = tuple()
    prediction_mask_field: str = "prediction_mask"
    expected_ndim: int = 2

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target = data_entry[self.target_field]
        prediction_mask = self._generate_prediction_mask(target)
        self.map_func(
            partial(self._truncate, mask=prediction_mask),  # noqa
            data_entry,
            self.truncate_fields,
            optional_fields=self.optional_truncate_fields,
        )
        data_entry[self.prediction_mask_field] = prediction_mask
        return data_entry

    def _generate_prediction_mask(
        self, target: Float[np.ndarray, "var time *feat"]
    ) -> Bool[np.ndarray, "var time"]:
        self.check_ndim("target", target, self.expected_ndim)
        var, time = target.shape[:2]
        prediction_mask = np.zeros((var, time), dtype=bool)
        prediction_mask[:, -self.mask_length :] = True
        return prediction_mask

    def _truncate(
        self,
        data_entry: dict[str, Any],
        field: str,
        mask: np.ndarray,
    ) -> np.ndarray | list[np.ndarray] | dict[str, np.ndarray]:
        arr: np.ndarray | list[np.ndarray] | dict[str, np.ndarray] = data_entry[field]
        if isinstance(arr, list):
            return [self._truncate_arr(a, mask) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.truncate_fields or k in self.optional_truncate_fields:
                    arr[k] = self._truncate_arr(v, mask)
            return arr
        return self._truncate_arr(arr, mask)

    @staticmethod
    def _truncate_arr(
        arr: Float[np.ndarray, "var time *feat"], mask: Bool[np.ndarray, "var time"]
    ) -> Float[np.ndarray, "var time-mask_len *feat"]:
        return arr[:, ~mask[0]]


@dataclass
class MaskedPredictionJump(MapFuncMixin, CheckArrNDimMixin, Transformation):
    target_field: str = "target"
    prediction_mask_field: str = "prediction_mask"
    predictiom_mask_field_jump: str = "prediction_mask_jump"
    #added by us

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target = data_entry[self.target_field]
        prediction_mask = data_entry[self.prediction_mask_field]
        sampled_indices_loss = data_entry['sampled_indices_loss']
        sampled_indices_perm = data_entry['sampled_indices_perm']
        indices_in_perm = [sampled_indices_perm.tolist().index(x) for x in sampled_indices_loss.tolist()]
        feature_names = data_entry['column_names'][data_entry['sampled_indices_loss']]
        effective_psz = data_entry['patch_size']
        _jump_mask = self._get_masked_jump(target[indices_in_perm], prediction_mask[indices_in_perm], feature_names, effective_psz)

        _jump_mask_final = np.zeros_like(target)
        _jump_mask_final_sub =_jump_mask_final[indices_in_perm] 
        try:
            _jump_mask_final_sub[prediction_mask[indices_in_perm]] = _jump_mask
        except:
            print("check_error.")
        _jump_mask_final[indices_in_perm] = _jump_mask_final_sub
        data_entry[self.predictiom_mask_field_jump] = _jump_mask_final 
        # print("checker")

        return data_entry


    
    def _get_freqs(self, freqs_list, mask_len):

        list_len = []

        for i, f in enumerate(freqs_list):
            if f=='D':
                fill_len = 0 #means we do not compute the loss on them #jump_mask[i].shape[0]
            elif f=="W":
                fill_len = 7
            elif f=="M":
                fill_len = 30
            elif f=="Q":
                fill_len = 30
            elif f=="A":
                fill_len = 30
            
            list_len.append(fill_len)
        
        return list_len

    def _get_masked_jump(self, arr, prediction_mask, feature_names, effective_psz=32):
        '''
        returns jump mask in the shape: (#features, total_len_masked_array (can be 32*N)) -> (-1, 32) reshaped
        '''

        freqs_list = [i.split('_')[0] for i in feature_names]
        no_features = len(freqs_list)

        NV, NP, PSZ = arr.shape
        patch_size =  arr.shape[-1]
        L = prediction_mask[0].sum()

        ext_mask = np.ones((NV,1), dtype=bool)
        prediction_mask_ext = np.concatenate([prediction_mask,ext_mask], axis=1)[:,1:]
        masked_values_with_buffer = arr[prediction_mask_ext].reshape(NV, -1, PSZ)
        #put back to original shapes to get real psz
        masked_values_with_buffer = masked_values_with_buffer[:,:,:effective_psz]
        masked_values_with_buffer = masked_values_with_buffer.reshape(NV, -1)[:, -(1 + L * effective_psz): ]

        list_nan_zero = self.get_first_non_zero_all_channels(masked_values_with_buffer)
        list_freqs = self._get_freqs(freqs_list, NV)

        jump_mask = np.zeros(shape=(NV, L*effective_psz))
        #TODO here make sure covariates have list freq 0
        for i in np.arange(NV):
            index_i = list_nan_zero[i]
            len_i = list_freqs[i]
            if index_i >0:
                jump_mask[i][index_i:index_i+len_i] = 1
        
        jump_mask = jump_mask.reshape(-1, effective_psz)

        padding_length =PSZ - effective_psz
        if padding_length:
            pad_width = [(0, 0), (0, PSZ - effective_psz)]
            jump_mask = np.pad(jump_mask, pad_width, mode='constant', constant_values=0)


        return jump_mask.reshape(-1, PSZ)
        # return list_nan_zero


    def get_first_non_zero_all_channels(self, masked_values_with_buffer):
        masked_values_with_buffer = masked_values_with_buffer - masked_values_with_buffer[:,[0]]
        list_first_nan_zeros = []
        for i in np.arange(masked_values_with_buffer.shape[0]):
            list_first_nan_zeros.append(self.find_first_non_zero(masked_values_with_buffer[i]))
        return list_first_nan_zeros

    def find_first_non_zero(self, arr):
        return np.nonzero(arr)[0][0] if arr.any() else -1

    def _generate_prediction_mask(
        self, target: Float[np.ndarray, "var time *feat"]
    ) -> Bool[np.ndarray, "var time"]:
        self.check_ndim("target", target, self.expected_ndim)
        var, time = target.shape[:2]
        prediction_mask = np.zeros((var, time), dtype=bool)
        mask_ratio = np.random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        mask_length = max(1, round(time * mask_ratio)) 
        prediction_mask[:, -mask_length:] = True
        return prediction_mask
