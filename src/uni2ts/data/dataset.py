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

from enum import Enum
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from uni2ts.common.sampler import Sampler, get_sampler
from uni2ts.common.typing import (
    BatchedData,
    BatchedDateTime,
    BatchedString,
    Data,
    FlattenedData,
    MultivarTimeSeries,
    UnivarTimeSeries,
)
from uni2ts.data.indexer import Indexer
from uni2ts.transform import Transformation
from uni2ts.modifications.exceptions import EmptyDataError


class SampleTimeSeriesType(Enum):
    """
    How to sample from the dataset.
    - none: do not sample, return the current index.
    - uniform: each time series sampled with equal probability
    - proportional: each time series sampled with probability proportional to it's length
    """

    NONE = "none"
    UNIFORM = "uniform"
    PROPORTIONAL = "proportional"


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
    ):
        """
        :param indexer: Underlying Indexer object
        :param transform: Transformation to apply to time series
        :param sample_time_series: defines how a time series is obtained from the dataset
        :param dataset_weight: multiplicative factor to apply to dataset size
        """
        self.indexer = indexer
        self.transform = transform
        self.sample_time_series = sample_time_series
        self.dataset_weight = dataset_weight

        if sample_time_series == SampleTimeSeriesType.NONE:
            self.probabilities = None
        elif sample_time_series == SampleTimeSeriesType.UNIFORM:
            self.probabilities = indexer.get_uniform_probabilities()
        elif sample_time_series == SampleTimeSeriesType.PROPORTIONAL:
            self.probabilities = indexer.get_proportional_probabilities()
        else:
            raise ValueError(f"Unknown sample type {sample_time_series}")

    def __getitem__(self, idx: int) -> dict[str, FlattenedData]:
        """
        Obtain a time series from the dataset, flatten
        :param idx: index of time series to retrieve. if sample_time_series is specified, this will be ignored.
        :return: transformed time series data
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        if self.sample_time_series != SampleTimeSeriesType.NONE:
            idx = np.random.choice(len(self.probabilities), p=self.probabilities)
        
        #dict_keys(['start', 'freq', 'item_id', 'target']) #shapes (),(),(),[ch, T]
        raw_data = self._get_data(idx)

        #dict_keys(['start', 'freq', 'item_id', 'target']) #shapes (),(),(), list = [T, ..., T] with ch elements 
        flatten_data = self._flatten_data(raw_data)
        try:
            assert len(flatten_data['target']) > 0, "Assertion failed: 'target' list is empty"
        except AssertionError as e:
            # print(e)
            # Optionally, re-raise the exception if you want it to propagate further
            raise
        # if flatten_data['target'] == []

        try:
            transformed_data = self.transform(flatten_data)
        except EmptyDataError:
            raise
        
        LtV, patch_size = transformed_data['target'].shape
        V = np.unique(transformed_data['variate_id']).shape[0]
        L = LtV//V
        L_missing = sum(transformed_data['prediction_mask'][:L])
        # print(
        #     f"idx {idx}, shape {transformed_data['target'].shape}, "
        #     f"channels {V}, L {L}, L missing {L_missing} "
        # )
        return transformed_data #HERE this is the transformation part #BREAKPOINT

    @property
    def num_ts(self) -> int:
        """
        Get the number of time series in the dataset
        """
        return len(self.indexer)

    def __len__(self) -> int:
        """
        Length is the number of time series multiplied by dataset_weight
        """
        return int(np.ceil(self.num_ts * self.dataset_weight))

    def _get_data(self, idx: int) -> dict[str, Data | BatchedData]:
        """
        Obtains time series from Indexer object
        """
        return self.indexer[idx % self.num_ts]

    @staticmethod
    def _flatten_data(data: dict[str, Data]) -> dict[str, FlattenedData]:
        """
        Convert time series type data into a list of univariate time series
        """
        return {
            k: (
                [v]
                if isinstance(v, UnivarTimeSeries)
                else list(v) if isinstance(v, MultivarTimeSeries) else v
            )
            for k, v in data.items()
        }


class MultiSampleTimeSeriesDataset(TimeSeriesDataset):
    """
    Samples multiple time series and stacks them into a single time series.
    Underlying dataset should have aligned time series, meaning same start and end dates.
    """

    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        max_ts: int,
        combine_fields: tuple[str, ...],
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
        sampler: Sampler = get_sampler("beta_binomial", a=2, b=5),
    ):
        """
        :param indexer: Underlying Indexer object
        :param transform: Transformation to apply to time series
        :param max_ts: maximum number of time series that can be stacked together
        :param combine_fields: fields which should be stacked
        :param sample_time_series: defines how a time series is obtained from the dataset
        :param dataset_weight: multiplicative factor to apply to dataset size
        :param sampler: how to sample the other time series
        """
        super().__init__(indexer, transform, sample_time_series, dataset_weight)
        self.max_ts = max_ts
        self.combine_fields = combine_fields
        self.sampler = sampler

    def _get_data(self, idx: int) -> dict[str, BatchedData]:
        n_series = self.sampler(min(self.num_ts, self.max_ts))
        choices = np.concatenate([np.arange(idx), np.arange(idx + 1, self.num_ts)])
        others = np.random.choice(choices, n_series - 1, replace=False)
        samples = self.indexer[np.concatenate([[idx], others])]
        return samples

    def _flatten_data(
        self, samples: dict[str, BatchedData]
    ) -> dict[str, FlattenedData]:
        for field in samples.keys():
            if field in self.combine_fields:
                item = samples[field]
                if isinstance(item, list) and isinstance(item[0], MultivarTimeSeries):
                    samples[field] = [
                        univar for sample in samples[field] for univar in sample
                    ]
            elif isinstance(samples[field], BatchedDateTime):
                samples[field] = np.asarray(samples[field][0])
            elif isinstance(samples[field], BatchedString):
                samples[field] = samples[field][0]
            else:
                raise AssertionError(
                    f"Field {field} not accounted for in {self.indexer} MultiSampleTimeSeriesDataset"
                )
        return samples


class EvalDataset(TimeSeriesDataset):
    """
    Dataset class for validation.
    Should be used in conjunction with Eval transformations.
    """

    def __init__(
        self,
        windows: int,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
    ):
        """
        :param windows: number of windows to perform evaluation on
        """
        super().__init__(
            indexer,
            transform,
            SampleTimeSeriesType.NONE,
            dataset_weight=windows,
        )

    def _get_data(self, idx: int) -> dict[str, Data]:
        window, idx = divmod(idx, self.num_ts)
        item = self.indexer[idx]
        item["window"] = window
        return item
