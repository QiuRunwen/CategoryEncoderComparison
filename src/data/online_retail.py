"""
Online Retail II
https://archive-beta.ics.uci.edu/dataset/502/online+retail+ii
https://www.openml.org/search?type=data&status=active&id=43368
TODO 聚合完以后没有cat_col
"""

import pandas as pd
import os
from zipfile import ZipFile
if __name__ == "__main__":
    import util # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用 

def load(data_dir='../../data', drop_useless=True, num_sample=None, verbose=False): 
    
    # if raw data
    # file_path = os.path.join('../../data', 'UCI/online_retail', 'online_retail_II.xlsx') 
    # # 使用with语句打开Excel文件并创建ExcelFile对象
    # with pd.ExcelFile(file_path) as xlsx:
    #     df1 = xlsx.parse(xlsx.sheet_names[0])
    #     df2 = xlsx.parse(xlsx.sheet_names[1])
    # df = pd.concat([df1,df2])
    # df.to_csv(file_path[:-4]+'csv', index=False)
    
    # 1. read/uncompress data
    file_path = os.path.join(data_dir, 'UCI/online_retail', 'online_retail_II.zip') 
    with ZipFile(file_path) as zf:
        with zf.open('online_retail_II.csv') as f:
            df = pd.read_csv(f,  parse_dates=['InvoiceDate'], date_format='%Y/%m/%d %H:%M:%S')
            
    # 1077371*8
    # InvoiceNo: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation. 
    # StockCode: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product. 
    # Description: Product (item) name. Nominal. 
    # Quantity: The quantities of each product (item) per transaction. Numeric.	
    # InvoiceDate: Invice date and time. Numeric. The day and time when a transaction was generated. 
    # UnitPrice: Unit price. Numeric. Product price per unit in sterling (Â£). 
    # CustomerID: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer. 
    # Country: Country name. Nominal. The name of the country where a customer resides.

    # 2. compute target
    # quantity_day_sale
    df = df[~df['Invoice'].str.startswith('c')] # only count the invoices which are not cancelled
    df.groupby
    y_col = 'quantity_day_sale'
    
    # 3. simple feature extraction

    
    
    
    # 4. convert numeric/categorical columns
    # cat_cols = ['ocean_proximity']
    # for col in cat_cols:
    #     if not pd.api.types.is_categorical_dtype(df[col]):
    #         df[col] = df[col].astype('category')

    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns
        
        # b. auto identified useless columns
        useless_cols_dict = util.find_useless_colum(df)
        df = util.drop_useless(df, useless_cols_dict, verbose=verbose)        
 
    # 6. sampling by class
    if num_sample is not None:
        df = util.sampling_by_class(df, y_col, num_sample)
        
        # remove categorical cols with too few samples_per_cat
        
    return df, y_col


if __name__ == "__main__":
    df,y_col = load(verbose=True)
    df.info()