
# Define multiple pairs of subsets and their corresponding sample sizes
# 'sales': {'features':[], 'freqs':['M']}
dataset_features = ['value', 'delta_value', 'log_value', 'delta_log_value']


subsets_1 = {
    # DAILY
    'ex_rate': {
        'features': ['EE58', 'FXSP'],
        'freqs': ['D']
    },
    'ex_rate_other': {
        'features': ['EE26'],
        'freqs': ['D']
    },
    # MONTHLY
    'money_stock': {
        'features': ['MNY1'],
        'freqs': ['M']
    },
    'interest_rate': {
        'features': ['POLR'],
        'freqs': ['M']
    },
    # QUARTERLY
    'gdp': {
        'features': ['GDPN'],
        'freqs': ['Q']
    },
    # PRICES (can be used for both monthly and quarterly)
    'prices': {
        'features': ['CPI', 'CPCH'],
        'freqs': ['M']
    },
    'prices_q': {
        'features': ['CPI', 'CPCH'],
        'freqs': ['Q']
    }
}

n_samples_1 = {
    'ex_rate': 1,
    'ex_rate_other': 1,
    'money_stock': 1,
    'interest_rate': 1,
    'gdp': 1,
    'prices': 1,
    'prices_q': 1
}


# Store all pairs in dictionaries
all_subsets = {
    '1': subsets_1,}

all_n_samples = {
    '1': n_samples_1,
}

all_covariates_filter = {
    '1': []}

# Sanity check: ensure corresponding subsets and n_samples have the same keys
for key in all_subsets:
    if key not in all_n_samples:
        raise ValueError(f"Missing n_samples for subset key: {key}")
    if set(all_subsets[key].keys()) != set(all_n_samples[key].keys()):
        raise ValueError(f"Keys mismatch between subsets and n_samples for key: {key}")