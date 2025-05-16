
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



# Pair 1
subsets_1 = {
    'subset1': ['Obs_value_1', 'Obs_value_2', 'Obs_value_3'],
    'subset2': ['Obs_value_4', 'Obs_value_5', 'Obs_value_6'],
    'subset3': ['Obs_value_7', 'Obs_value_8', 'Obs_value_9', 'Obs_value_10']
}

n_samples_1 = {
    'subset1': 2,
    'subset2': 2,
    'subset3': 2
}

# Pair 2
subsets_2 = {
    'subsetA': ['Obs_value_1', 'Obs_value_2'],
    'subsetB': ['Obs_value_3', 'Obs_value_4', 'Obs_value_5'],
    'subsetC': ['Obs_value_6', 'Obs_value_7', 'Obs_value_8', 'Obs_value_9', 'Obs_value_10']
}

n_samples_2 = {
    'subsetA': 1,
    'subsetB': 2,
    'subsetC': 3
}



# Pair 2
subsets_2 = {
    'subsetA': ['Obs_value_1', 'Obs_value_2'],
    'subsetB': ['Obs_value_3', 'Obs_value_4', 'Obs_value_5'],
    'subsetC': ['Obs_value_6', 'Obs_value_7', 'Obs_value_8', 'Obs_value_9', 'Obs_value_10']
}

n_samples_2 = {
    'subsetA': 1,
    'subsetB': 2,
    'subsetC': 3
}


subsets_3 = {
    'ex_rate': {'features': ['QBCA'], 'freqs' : ['D']},
    'ex_rate_other': {'features': ['QREA', 'QRFA'], 'freqs' : ['D']},
    'monthly': {'features': ['HBAA'], 'freqs' : ['M']},
}

n_samples_3 = {
    'ex_rate': 1,
    'ex_rate_other': 0,
    'monthly': 1
}



subsets_3 = {
    'ex_rate': {'features': ['QBCA'], 'freqs' : ['D']},
    'ex_rate_other': {'features': ['QREA', 'QRFA'], 'freqs' : ['D']},
    'monthly': {'features': ['HBAA'], 'freqs' : ['M']},
}

n_samples_3 = {
    'ex_rate': 1,
    'ex_rate_other': 1,
    'monthly': 1
}

subsets_4 = {
    #DAILY
    'ex_rate': {'features': ['QBCA']
                    , 'freqs' : ['D']},
    'ex_rate_other': {'features': ['QREA', 'QRFA']
                    , 'freqs' : ['D']},
    #MONTHLY
    'money_stock': {'features': ["AABA", "ABBA", "ABBB", "ABBC", "ABBD", "ABHA", "ABHB", "ABNA", "ABNB", "ABNC", "ABND", "ABUA", "ABUB", "ABUC", "ABUD", "ACBA", "ACBB", "ACBC", 
                        "ACBD", "ACHA", "ACHB", "ACKA", "ACKB", "ACKC", "ACKD", "ACPA", "BAAA", "BBBB"]
                    ,'freqs' : ['M']},
    'interest_rate': {'features':["HBAA", "HBBA", "HBDA", "HBFA", "HBGA", "HBHA", "HBJA", "HEBA", "HEEA", "HEFA", "HEHA", "HELA", "HLBA", "HLEA", "HLHA", "HLIA", "HLJA", "HLLA", "HLMA"]
                    ,'freqs' : ['M']},
    'bonds': {'features':["HGBA", "HGCA", "HGEA", "HGHA", "HGLA", "HGPA"]
                    , 'freqs':['M']},
    'production_sales': {'features':["SBBA", "SBBB", "SBBC", "SBNA", "SBNB", "SBNC", "SEBA", "SEBB", "SEBC", "SKBA", "SKBB", "TLBA",
                                         "TLBB", "TLBC", "TLBD", "TQBA", "TQBB", "TTBB", "TTNA", "TTNB", "TTRA", "TTRB"]
                    , 'freqs':['M']},
    'labour': {'features':["UBNA", "UBNB", "UDHB", "UIHA", "UIHB", "UKBB", "UMBB", "UQBA", "UQNA", "UQNB", "UQNC"]
                    , 'freqs':['M']},
    'prices': {'features':["VBBA", "VBBB", "VBBC", "VBNA", "VBNB", "VCAA", "VEBA", "VEBB", "VEBC", "VEBD", "VEDA", 
                    "VEDC", "VEFA", "VEFB", "VEFC", "VEFD", "VEHA", "VEHB", "VEHC", "VFBC", "VHBA", "VHBB", "VHDA", "VHDC", "VHJA", "VHJC", "VJLA", "VJLB"]
                    , 'freqs':['M']},
    'wages': {'features':["VNBA", "VNNA"], 'freqs':['M']},
    #QUARTERLY
    'gdp':  {'features':["RBAA", "RBAB", "RBBA", "RBBB", "RBGA", "RBGB"]
                    , 'freqs':['Q']},
    'consumption': {'features':["RCGA", "RCGB", "RCLA", "RCLB"]
                    , 'freqs':['Q']},
    'investments': {'features':['RDAA', "RDAB", "RDBA", "RDBB", "RDCA", "RDCB", "RDDA", "RDDB", "RDGA", "RDGB", "RDLA", "RDLB"]
                    , 'freqs':['Q']},
    'price_deflator': {'features':[ "RNGA", "RNGB", "RNGC", "RNGD"]
                    , 'freqs':['Q']},
    'labour_q': {'features':["UBNA", "UBNB", "UIHA", "UIHB", "UKBB", "UMBB", "UMDB", "UQBA", "UQBB", "UQBC", "UQNA", "UQNB"]
                    , 'freqs':['Q']},
    'prices_q':  {'features':['VEBA','VEBC']
                    , 'freqs':['Q']},
    'wage_q': {'features':['VLBA','VLBB','VNBA','VPBA','VPBB','VPNA','VPNB']
                    , 'freqs':['Q']},

    #COVARIATES
    'commodity':{'features':['ALUM03', 'COPP02DL', 'PET03', 'RICE01', 'WHEAT2', 'WOOL04', 'ZINC03'], 'freqs':['M']},
    # #ANNUAL
    # 'consumption_a': {'features':['RCGA','RCLA']
    #                 , 'freqs':['A']},
    # 'labour_a': {'features':['UBNA','UIHA','UIHB','UQBA','UQBC','UQNA']
    #                 , 'freqs':['A']},
    # 'prices_a': {'features':['VEBA','VEBC','VHDA','VHDC','VHJA']
    #                 , 'freqs':['A']},
    # 'wage_a': {'features':['VPBA','VPBC','VPNA']
    #                 , 'freqs':['A']}
}




n_samples_4 = {
 'ex_rate': 1,
 'ex_rate_other': 1,
 'money_stock': 1,
 'interest_rate': 1,
 'bonds': 1,
 'production_sales': 1,
 'labour': 1,
 'prices': 1,
 'wages': 1,
 'gdp': 1,
 'consumption': 1,
 'investments': 1,
 'price_deflator': 1,
 'labour_q': 1,
 'prices_q': 1,
 'wage_q': 1,
 'commodity':3
#  'consumption_a': 1,
#  'labour_a': 1,
#  'prices_a': 1,
#  'wage_a': 1
 }



# subsets_5 = {    #QUARTERLY
#     'gdp':  {'features':["RBGA", "RBGB"]
#                     , 'freqs':['Q']},
#     'inflation': {'features':["VEBA"]
#                     , 'freqs':['Q', 'M']},
#     'unemployment': {'features':["UIHA", "UIHB"]
#                     , 'freqs':['Q', 'M']}
# }


subsets_5={
    'ex_rate': {'features': ['QBCA']
                    , 'freqs' : ['D']},
    'eff_exc': {'features': ['QREA', 'QRFA'], 'freqs': ['D']},

    'interest_rate': {'features':["HBAA", "HBBA", "HBDA", "HBFA", "HBGA", "HBHA", "HBJA", "HEBA", "HEEA", "HEFA", "HEHA", "HELA", "HLBA", "HLEA", "HLHA", "HLIA", "HLJA", "HLLA", "HLMA"]
                    ,'freqs' : ['M']},

    'ind_production': {'features': ["SBBA", "SBBB", "SBBC", "SBNA", "SBNB", "SBNC", "SEBA", "SEBB", "SEBC", "SKBA", "SKBB"], 'freqs': ['M']},

    'emp': {'features': ['UDHB', 'UKKB'], 'freqs': ['M', 'Q']},

    'ppi': {'features': ["VBBA", "VBBB", "VBBC", "VBNA", "VBNB", "VCAA",], 'freqs': ['M']},

    'bonds': {'features':["HGBA", "HGCA", "HGEA", "HGHA", "HGLA", "HGPA"]
                    , 'freqs':['M']},

    'investments': {'features':['RDAA', "RDAB", "RDBA", "RDBB", "RDCA", "RDCB"]
                    , 'freqs':['Q']},
    
    'consumption': {'features':["RCGA", "RCGB"]
                    , 'freqs':['Q']},

    'wages': {'features': ["VBNA", "VNNA"], 'freqs':['M']},


    'gdp': {'features': ['RBGA', 'RBGB'], 'freqs': ['Q']},
    
    'unemployment': {'features': ['UIHA', 'UIHB'], 'freqs': ['M']},

    'inflation':  {'features': ['VEBA'], 'freqs': ['M', 'Q']},

    'commodity':{'features':['PET03'], 'freqs':['M']},

    }


n_samples_5 = {
    'ex_rate':1,
    'eff_exc':1,
    'interest_rate':1,
    'ind_production':1,
    'emp':1,
    'ppi':1,
    'bonds':1,
    'investments':1,
    'consumption':1,
    'wages':1,
    'gdp':1,
    'unemployment':1,
    'inflation':1,
    'commodity':1
}

# Store all pairs in dictionaries
all_subsets = {
    '1': subsets_1,
    '2': subsets_2,
    '3': subsets_3,
    '4': subsets_4,
    '5': subsets_5
}

all_n_samples = {
    '1': n_samples_1,
    '2': n_samples_2,
    '3': n_samples_3,
    '4': n_samples_4,
    '5': n_samples_5
}

all_covariates_filter = {
    '4': ['commodity'],
    '5': ['ex_rate',
    'eff_exc',
    'interest_rate',
    'ind_production',
    'emp',
    'ppi',
    'bonds',
    'investments',
    'consumption',
    'wages',
    'commodity']
}

# Sanity check: ensure corresponding subsets and n_samples have the same keys
for key in all_subsets:
    if key not in all_n_samples:
        raise ValueError(f"Missing n_samples for subset key: {key}")
    if set(all_subsets[key].keys()) != set(all_n_samples[key].keys()):
        raise ValueError(f"Keys mismatch between subsets and n_samples for key: {key}")