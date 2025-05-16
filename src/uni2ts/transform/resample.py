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
STORE_LOGS = dict()

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from uni2ts.common.sampler import Sampler, get_sampler
from uni2ts.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin

from uni2ts.modifications.sample_per_subset import FeatureSampler, generate_freq_feature_dict, get_parts_before_second_underscore, count_keys
from uni2ts.modifications.scenerios import all_subsets, all_n_samples, dataset_features, all_covariates_filter
from uni2ts.modifications.exceptions import EmptyDataError

import random

@dataclass
class SampleDimension(
    CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin, Transformation
):
    max_dim: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    sampler: Sampler = get_sampler("uniform")
    per_subset: bool = False
    subsets_id: str = None
    split: str = 'train'
    use_features: tuple[str, ...] =  None

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        #HERE we are getting total channels in the data
        total_field_dim = sum(
            self.collect_func_list(
                self._get_dim,
                data_entry,
                self.fields,
                optional_fields=self.optional_fields,
            )
        )

        self.use_features = list(self.use_features)
        assert all(element in dataset_features for element in self.use_features), "Not all elements of self.use_features are in dataset_features"
        
        #HERE we are sampling per channel
        self.map_func(
            partial(self._process, total_field_dim=total_field_dim),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )

        try:
            assert len(data_entry['target']) > 0, "Assertion failed: 'target' is empty"
            # assert len(data_entry['sampled_indices_loss']) > 0, "Assertion failed: 'sampled_indices_loss' is empty"
        except AssertionError as e:
            if data_entry['country_name'] in STORE_LOGS:
                STORE_LOGS[data_entry['country_name']] +=1
            else:
                STORE_LOGS[data_entry['country_name']] = 1
            raise

        try:
            assert len(data_entry['sampled_indices_loss']) > 0, "Assertion failed: 'sampled_indices_loss' is empty"
        except AssertionError as e:
            raise EmptyDataError

        return data_entry

    def _get_dim(self, data_entry: dict[str, Any], field: str) -> int:
        self.check_ndim(field, data_entry[field], 2)
        return len(data_entry[field])

    def _process(
        self, data_entry: dict[str, Any], field: str, total_field_dim: int
    ) -> list[UnivarTimeSeries]:
        arr: list[UnivarTimeSeries] = data_entry[field]
        
        field_max_dim = (self.max_dim * len(arr)) // total_field_dim
        if not self.per_subset or field == "past_feat_dynamic_real":
            rand_idx = np.random.permutation(len(arr))
            self.sampler = get_sampler("uniform")
            n = self.sampler(min(len(arr), field_max_dim))     
            return [arr[idx] for idx in rand_idx[:n]]

        else:
            print(self.subsets_id)
            print(all_subsets)
            subsets = all_subsets[self.subsets_id]
            n_samples = all_n_samples[self.subsets_id]
            covariates_filter = all_covariates_filter[self.subsets_id]
            feature_names = data_entry['feature_names']
            feature_names_cov = data_entry['feature_names_cov'] 
            feature_names_all = np.concatenate((feature_names,feature_names_cov), axis=0)
            self.sampler = FeatureSampler(max_dim = len(arr), sampler = lambda x: x)

            use_feature_indexes = [dataset_features.index(feature) for feature in self.use_features] #get indexes of features


            #if there is a maximum number (field_max_dim) of variates to sample 
            field_max_dim = field_max_dim //len(self.use_features)


            sampled_indices_features = self.sampler._process(feature_names_all, total_field_dim=None, subsets=subsets, n_samples=n_samples)
            sampled_indices_features = random.sample(sampled_indices_features, min(len(sampled_indices_features), field_max_dim))


            #HERE you can check what we have sample from the features print(feature_names[sampled_indices_features])
            sampled_indices = [i*len(dataset_features) + np.array(use_feature_indexes) for i in sampled_indices_features] #find the real index in dataframe
            # sampled_indices = np.concatenate(sampled_indices)
            sampled_indices = np.concatenate(sampled_indices) if sampled_indices else np.array([])

            
            #ALWAYS PERMUTE THE INDICES RANDOMLY #SO THAT DATA IS PERMUTED EACH ITERATION
            sampled_indices_perm = np.random.permutation(sampled_indices)
            if self.split == 'val':
                sampled_indices_perm = sampled_indices
            data_entry['sampled_indices'] = sampled_indices
            data_entry['sampled_indices_perm'] = sampled_indices_perm
            data_entry['selected_features'] = feature_names_all[sampled_indices_features] #HERE sampled indices
            
            #in sampled_indices_loss we do not want to have covariates! #we need to add here more 
            cov_features = []
            if len(covariates_filter)>0:
                for i in covariates_filter:
                    cov_features = cov_features+ subsets[i]['features']

            sampled_indices_features_filtered = [i for i in sampled_indices if data_entry['column_names'][i].split('_')[1] not in cov_features]
            n_consider_feat = len(use_feature_indexes) if self.split =='train' else 1 
            data_entry['sampled_indices_loss'] = self.random_sample_from_groups(sampled_indices_features_filtered, len_sub=len(use_feature_indexes), n_consider_feat=n_consider_feat)

            # data_entry['sampled_indices_loss'] = self.random_sample_from_groups(sampled_indices, len_sub= len(dataset_features))
            # get columns sampled for debugging purposes
            # print('Split:', self.split)
            # print('Country Name:',data_entry['country_name'],'Start:', data_entry['start'])             
            # print("Sampled indices:", data_entry['column_names'][data_entry['sampled_indices']])
            # print("Sampled indices loss:", data_entry['column_names'][data_entry['sampled_indices_loss']])

            try:
                assert len(data_entry['sampled_indices_loss']) > 0, "Assertion failed: 'sampled_indices_loss' is empty"
            except AssertionError as e:
                raise EmptyDataError

            return [arr[idx] for idx in sampled_indices_perm.tolist()]

    def random_sample_from_groups(self, arr, len_sub, n_consider_feat=4):
        """
        Randomly sample one number from every group of four elements in the input array.

        Parameters:
        arr (numpy.ndarray): Input array of numbers.

        Returns:
        numpy.ndarray: Array containing one randomly sampled number from each group of four elements.
        """
        # Initialize an empty list to store the samples
        import random
        sampled_array = []

        # Iterate through the array in steps of 4
        for i in range(0, len(arr), len_sub):
            # Get the current group of four elements
            group = arr[i:i+n_consider_feat]
            # Randomly select one element from the group
            sampled_element = random.choice(group)
            # Append the sampled element to the sampled_array list
            sampled_array.append(sampled_element)

        # Convert the sampled_array list back to a numpy array
        return np.array(sampled_array)

# # Example usage
# arr = np.array([244, 245, 246, 247, 52, 53, 54, 55])
# sampled_array = random_sample_from_groups(arr)
# print(sampled_array)

@dataclass
class Subsample(Transformation):  # just take every n-th element
    fields: tuple[str, ...] = ("target", "past_feat_dynamic_real")

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        pass


class GaussianFilterSubsample(
    Subsample
):  # blur using gaussian filter before subsampling
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # gaussian filter
        return super()(data_entry)


class Downsample(Transformation):  # aggregate
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        pass


class Upsample(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        pass
