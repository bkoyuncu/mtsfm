
import numpy as np
from typing import List, Dict
from collections import defaultdict

def count_keys(features_array, freq_feature_dict):
    # Initialize a defaultdict to count occurrences
    counts = defaultdict(int)

    # Count the occurrences of each key
    for feature in features_array:
        if feature in freq_feature_dict:
            key = freq_feature_dict[feature]
            counts[key] += 1

    # Convert defaultdict to a regular dictionary
    return dict(counts)


def get_parts_before_second_underscore(features_names):
    parts_before_second_underscore = []
    for feature in features_names:
        part_before_second_underscore = '_'.join(feature.split('_')[:2])
        parts_before_second_underscore.append(part_before_second_underscore)
    return np.array(parts_before_second_underscore)


def generate_freq_feature_dict(subsets):
    freq_feature_dict = {}
    
    for key, value in subsets.items():
        freqs = value['freqs']
        features = value['features']
        for freq in freqs:
            for feature in features:
                freq_feature = f"{freq}_{feature}"
                freq_feature_dict[freq_feature] = key
                
    return freq_feature_dict

    
def filter_names_by_keywords(names: np.ndarray, keywords: List[str]) -> List[str]:
    return [name for name in names if any(keyword in name for keyword in keywords)]

def get_indices_of_filtered_names(names: np.ndarray, filtered_names: List[str]) -> List[int]:
    idx_tuples =  [(idx,filtered_names.index(name)) for idx, name in enumerate(names) if name in filtered_names]
    # Sort the tuples by their second value
    sorted_idx_tuples = sorted(idx_tuples, key=lambda x: x[1])
    # Extract the first values from the sorted tuples, return indices in the given order
    return  [x[0] for x in sorted_idx_tuples]

# def custom_sort(names: List[str], order: List[str]) -> List[str]:
#     # Create a dictionary to map each prefix to its position in the order list
#     order_dict = {prefix: index for index, prefix in enumerate(order)}
    
#     # Define a sorting key function that uses the order dictionary
#     def sort_key(name: str) -> int:
#         # Extract the prefix from the name (assuming the prefix is the first character)
#         prefix = name.split('_')[0]
#         return order_dict.get(prefix, len(order))  # Default to len(order) if prefix not found

#     # Sort the names using the custom sorting key
#     sorted_names = sorted(names, key=sort_key)
#     return sorted_names

def custom_sort(names: List[str], order: List[str]) -> List[str]:
    # Create a dictionary to map each prefix to its position in the order list
    order_dict = {prefix: index for index, prefix in enumerate(order)}
    
    # Define a sorting key function that uses the order dictionary
    def sort_key(name: str) -> int:
        # Extract the prefix from the name (assuming the prefix is the first character)
        prefix = name.split('_')[0]
        return order_dict.get(prefix, len(order))  # Default to len(order) if prefix not found

    # Filter the names to only include those with a prefix found in the order list
    filtered_names = [name for name in names if name.split('_')[0] in order_dict]
    
    # Sort the filtered names using the custom sorting key
    sorted_names = sorted(filtered_names, key=sort_key)
    
    return sorted_names

class Feature:
    def __init__(self, name: str):
        self.name = name

class FeatureSampler:
    def __init__(self, max_dim: int, sampler):
        self.max_dim = max_dim
        self.sampler = sampler

    def _process(
        self, features: List[Feature], total_field_dim: int,
        subsets: Dict[str, List[str]], n_samples: Dict[str, int]
    ) -> List[int]:
        # Create a mapping of feature names to their indices in the entire list
        sampled_features_indices = []
        sampled_features = []
        # Iterate over each subset and its corresponding number of samples
        for subset_name, subset in subsets.items():
            n = n_samples[subset_name]
            # Filter array based on the subset

            filtered_names = filter_names_by_keywords(features, subset['features'])
            sorted_filtered_names = custom_sort(filtered_names, subset['freqs'])
            filtered_indices = get_indices_of_filtered_names(features, sorted_filtered_names)

            # Sample from the subset
            no_samples = min(n, len(filtered_indices))

            if no_samples>0:
                sampled_subset = np.random.choice(filtered_indices, no_samples, replace=False) if no_samples>0 else np.array([])
                # Add sampled features to the list
                if len(sampled_subset)>0:
                    sampled_features_indices.extend(sampled_subset)
                    sampled_features.extend(features[sampled_subset])
            # else:
            # print(f'Cannot sample from {subset_name}') #BREAKPOINT
                

        # Convert sampled features back to their indices in the entire list
        # sampled_indices = [feature_name_to_index[feature_name] for feature_name in sampled_features]
        
        return sampled_features_indices

# # Example usage
# sampler = FeatureSampler(max_dim=10, sampler=lambda x: x)
# features = [Feature(f'Obs_value_{i}') for i in range(1, 11)]

# # Use the first pair of subsets and sample sizes
# sampled_indices_1 = sampler._process(features, total_field_dim=10, subsets=all_subsets['pair1'], n_samples=all_n_samples['pair1'])
# print("Sampled indices for pair 1:", sampled_indices_1)

# # Use the second pair of subsets and sample sizes
# sampled_indices_2 = sampler._process(features, total_field_dim=10, subsets=all_subsets['pair2'], n_samples=all_n_samples['pair2'])
# print("Sampled indices for pair 2:", sampled_indices_2)

# #====FIRST IMPLEMENTATION

# import numpy as np
# from typing import List

# class Feature:
#     def __init__(self, name: str):
#         self.name = name

# class FeatureSampler:
#     def __init__(self, max_dim: int, sampler):
#         self.max_dim = max_dim
#         self.sampler = sampler

#     def _process(
#         self, features: List[Feature], total_field_dim: int,
#         subset1: List[str], subset2: List[str], subset3: List[str],
#         n1: int, n2: int, n3: int
#     ) -> List[int]:
#         # Create a mapping of feature names to their indices in the entire list
#         feature_name_to_index = {feature_name: idx for idx, feature_name in enumerate(features)}
        
#         # Filter arrays based on the subsets
#         subset1_arr = [item for item in features if item in subset1]
#         subset2_arr = [item for item in features if item in subset2]
#         subset3_arr = [item for item in features if item in subset3]

#         # Sample from each subset
#         sampled_subset1 = np.random.choice(subset1_arr, min(n1, len(subset1_arr)), replace=False)
#         sampled_subset2 = np.random.choice(subset2_arr, min(n2, len(subset2_arr)), replace=False)
#         sampled_subset3 = np.random.choice(subset3_arr, min(n3, len(subset3_arr)), replace=False)

#         # Combine sampled features
#         sampled_features = list(sampled_subset1) + list(sampled_subset2) + list(sampled_subset3)

#         # Convert sampled features back to their indices in the entire list
#         sampled_indices = [feature_name_to_index[feature_name] for feature_name in sampled_features]
        
#         return sampled_indices

# # # Example usage
# # sampler = FeatureSampler(max_dim=10, sampler=lambda x: x)
# # features = [Feature(f'Obs_value_{i}') for i in range(1, 11)]

# # subset1 = ['Obs_value_1', 'Obs_value_2', 'Obs_value_3']
# # subset2 = ['Obs_value_4', 'Obs_value_5', 'Obs_value_6']
# # subset3 = ['Obs_value_7', 'Obs_value_8', 'Obs_value_9', 'Obs_value_10']

# # sampled_indices = sampler._process(features, total_field_dim=10,
# #                                    subset1=subset1, subset2=subset2, subset3=subset3,
# #                                    n1=2, n2=2, n3=2)

# # print("Sampled indices:", sampled_indices)