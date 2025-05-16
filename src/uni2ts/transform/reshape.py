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
from typing import Any, Optional

import numpy as np
from einops import pack

from ._base import Transformation
from ._mixin import CollectFuncMixin, MapFuncMixin


@dataclass
class SequencifyField(Transformation):
    field: str
    axis: int = 0
    target_field: str = "target"
    target_axis: int = 0
    #HERE repeats 'patch_size' as data_entry['target].shape[0], (ch * #patches) 
    #so data_entry['patch_size'] = [patch_size]* (ch * #patches)
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        data_entry[self.field] = data_entry[self.field].repeat(
            data_entry[self.target_field].shape[self.target_axis], axis=self.axis
        )
        return data_entry


@dataclass
class PackFields(CollectFuncMixin, Transformation):
    output_field: str
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    feat: bool = False
    #HERE collects every field given into 'target' or 'past_feat_dynamic_real' [fields]
    def __post_init__(self):
        self.pack_str: str = "* time feat" if self.feat else "* time"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        fields = self.collect_func_list(
            self.pop_field,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        if len(fields) > 0:
            output_field = pack(fields, self.pack_str)[0]
            data_entry |= {self.output_field: output_field}
        return data_entry

    @staticmethod
    def pop_field(data_entry: dict[str, Any], field: str) -> Any:
        return np.asarray(data_entry.pop(field))


@dataclass
class FlatPackFields(CollectFuncMixin, Transformation):
    output_field: str
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    feat: bool = False
    #HERE gets data_entry['target'] and pack it from (ch, #patches, patch_length) to (-1, patch_length)
    def __post_init__(self):
        self.pack_str: str = "* feat" if self.feat else "*"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        fields = self.collect_func_list(
            self.pop_field,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        if len(fields) > 0:
            output_field = pack(fields, self.pack_str)[0]
            data_entry |= {self.output_field: output_field}
        return data_entry

    @staticmethod
    def pop_field(data_entry: dict[str, Any], field: str) -> Any:
        return np.asarray(data_entry.pop(field))


@dataclass
class PackCollection(Transformation):
    field: str
    feat: bool = False

    def __post_init__(self):
        self.pack_str: str = "* time feat" if self.feat else "* time"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        collection = data_entry[self.field]
        if isinstance(collection, dict):
            collection = list(collection.values())
        data_entry[self.field] = pack(collection, self.pack_str)[0]
        return data_entry


@dataclass
class FlatPackCollection(Transformation):
    field: str
    feat: bool = False
    #HERE gets field values and flatten data_entry[self.field]['target'] from  shape (?) (ch, #patches) puts to data_entty[self.field] w shapes (ch * #patches)
    #observed mask from (ch,#patches,patch_size) to (ch*#patches, patch_size) 
    def __post_init__(self):
        self.pack_str: str = "* feat" if self.feat else "*"
    
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        collection = data_entry[self.field]
        if isinstance(collection, dict):
            collection = list(collection.values())
        data_entry[self.field] = pack(collection, self.pack_str)[0]
        return data_entry


@dataclass
class unpackPseudo(Transformation):
    field: str
    feat: bool = False
    #get the value out of it
    def __post_init__(self):
        self.pack_str: str = "* feat" if self.feat else "*"
    
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        collection = data_entry[self.field]
        # no_feat = collection[0].shape[0]
        data_entry[self.field] = {'target': collection[0]}
        return data_entry



@dataclass
class Transpose(MapFuncMixin, Transformation):
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    axes: Optional[tuple[int, ...]] = None

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.map_func(
            self.transpose,
            data_entry,
            fields=self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def transpose(self, data_entry: dict[str, Any], field: str) -> Any:
        out = data_entry[field].transpose(self.axes)
        return out
