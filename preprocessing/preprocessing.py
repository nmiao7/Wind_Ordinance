import pandas as pd

# Path to your data (run the code from the project level)
data_path = 'data/merged_data.csv'
data = pd.read_csv(data_path)

# Drop unneeded columns
data.drop(['case_id', 'faa_ors', 'faa_asn', 'usgs_pr_id', 't_fips', 't_cap',
           'p_name', 't_conf_loc', 't_img_date', 't_img_srce', 'retrofit',
           'retrofit_year', 't_conf_atr', 't_manu', 't_model'], axis=1, inplace=True)


# Printing DataFrame head and columns
print(data.head())
print(data.columns)

project_data = data.groupby(['t_state', 't_county', 'eia_id']).agg({
    'p_year': 'max',
    'p_tnum': 'max',  
    'p_cap': 'max',
    't_hh': 'mean',
    't_rd': 'mean',
    't_rsa': 'mean',
    't_ttlh': 'mean', 
    'xlong': 'mean', 
    'ylat': 'mean',  
    'ordinance': 'max'
}).reset_index()

print(project_data.head())
print(project_data.columns)

project_data.to_csv('data/merged_plevel_data.csv', index=False)

def weighted_average(values, weights):
    return (values * weights).sum() / weights.sum()

county_data = project_data.groupby(['t_state', 't_county', 'ordinance']).agg({
    'p_tnum': 'sum',
    'p_cap': 'sum',
    't_hh': lambda x: weighted_average(x, project_data.loc[x.index, 'p_tnum']),
    't_rd': lambda x: weighted_average(x, project_data.loc[x.index, 'p_tnum']),
    't_rsa': lambda x: weighted_average(x, project_data.loc[x.index, 'p_tnum']),
    't_ttlh': lambda x: weighted_average(x, project_data.loc[x.index, 'p_tnum']),
    'xlong': lambda x: weighted_average(x, project_data.loc[x.index, 'p_tnum']),
    'ylat': lambda x: weighted_average(x, project_data.loc[x.index, 'p_tnum']),
}).reset_index()

print(county_data.head())
print(county_data.columns)

county_data.to_csv('data/merged_clevel_data.csv', index=False)