"""
Download data from OpenML, given the id of dataset.
https://www.openml.org/

"""

import openml
import pandas as pd
import os
# df = openml.datasets.list_datasets(output_format="dataframe")
# df[(df["NumberOfClasses"]==0)
# &(df["NumberOfSymbolicFeatures"]>=1)
# &(df["NumberOfInstances"].between(1000,50000))
# &(df["NumberOfFeatures"].between(10,20))
# ][['did','name','NumberOfClasses', 'NumberOfFeatures','NumberOfSymbolicFeatures', 'NumberOfInstances']]

d_dataset_id = {
    'diamonds':42225,
    'nyc-taxi-green-dec-2016':42729,
    'particulate-matter-ukair-2017':42207,
    'medical_charges':41444,
    'KDDCup09_upselling':1114,
    'kick':41162,
    "churn": 41283,
    "CPMP2015":41700,
    "cholesterol":204,
    "moneyball":41021,
    "socmob":541,
    "chscase_foot":703,
    "CPS1988":43963,
    # "compass":44053, # binary
}

for name, ds_id in d_dataset_id.items():
    os.makedirs(name, exist_ok=True)
    fp = os.path.join(name, f'{name}.csv')
    if os.path.exists(fp):
        print(f'`{name}` already exists. Skip.')
        continue
    # 在`C:\Users\Run\.openml\org\openml\www\datasets\`下生成数据集
    dataset = openml.datasets.get_dataset(ds_id, download_data=False) 
    
    # X - An array/dataframe where each row represents one example with
    # the corresponding feature values.
    # y - the classes for each example
    # categorical_indicator - an array that indicates which feature is categorical
    # attribute_names - the names of the features for the examples (X) and
    # target feature (y)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    df = pd.concat([X, y], axis=1)
    df.to_csv(fp, index=False)
    print(f'`{name}` has been saved in `{os.path.abspath(fp)}`')