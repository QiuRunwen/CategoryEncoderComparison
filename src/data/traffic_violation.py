
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 10:02:40 2022

@author: YingFu
1. kaggle
https://www.kaggle.com/rounak041993/traffic-violations-in-maryland-county
2. a presentation of data preprocessing 
https://prezi.com/mzv9hoisr-z1/traffic-violations-in-montgomery-county/
"""

import pandas as pd
import os
from zipfile import ZipFile
if __name__ == "__main__":
    import util # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用 


def load(data_dir='../../data', drop_useless=True, num_sample=400000, verbose=False): 

    # 1. read/uncompress data
    with ZipFile(os.path.join(data_dir, 'kaggle/traffic_violations/Traffic_Violations.csv.zip')) as zipfile:
        with zipfile.open('Traffic_Violations.csv') as f:
            df = pd.read_csv(f,low_memory=False)  # read_csv 默认 low_memory 是True，第 26 列有很多空值，所以会认为这列有 mixed types
            # mixed_type_columns = df.iloc[:,26]
            # print(mixed_type_columns.value_counts(dropna=False)) 
            # Transportation Article    1214130
            # NaN                         65169
            # Maryland Rules              13100
    
    # print(df.info())
    # RangeIndex: 1292399 entries, 0 to 1292398
    # Data columns (total 35 columns):
    #  #   Column                   Non-Null Count    Dtype  
    # ---  ------                   --------------    -----  
    #  0   Date Of Stop             1292399 non-null  object  违章发生的日期，如 02/14/2014
    #  1   Time Of Stop             1292399 non-null  object  违章发生的时间，如 20:10:00
    #  2   Agency                   1292399 non-null  object  违章处理警局，unique value 
    #  3   SubAgency                1292389 non-null  object  违章处理的下属警局，unique value 
    #  4   Description              1292390 non-null  object  违章的具体文本描述
    #  5   Location                 1292397 non-null  object  违章的位置，文本
    #  6   Latitude                 1197045 non-null  float64 违章位置的纬度
    #  7   Longitude                1197045 non-null  float64 违章位置的经度
    #  8   Accident                 1292399 non-null  object  是否引发事故
    #  9   Belts                    1292399 non-null  object  是否涉及安全带的使用
    #  10  Personal Injury          1292399 non-null  object  是否有人员受伤
    #  11  Property Damage          1292399 non-null  object  是否有财产损失
    #  12  Fatal                    1292399 non-null  object  是否有人员致死
    #  13  Commercial License       1292399 non-null  object  是否为公司牌照
    #  14  HAZMAT                   1292399 non-null  object  事故中是否涉及危险品
    #  15  Commercial Vehicle       1292399 non-null  object  违章车是否为商务车
    #  16  Alcohol                  1292399 non-null  object  是否涉及酒驾
    #  17  Work Zone                1292399 non-null  object  违章是否发生在工作区
    #  18  State                    1292340 non-null  object  是否为国家牌照的公务车
    #  19  VehicleType              1292399 non-null  object  车辆类型(Automobile,Light Duty Truck)
    #  20  Year                     1284325 non-null  float64 车辆出产日期
    #  21  Make                     1292342 non-null  object  车辆品牌
    #  22  Model                    1292212 non-null  object  车的类型 (SUV,TK)
    #  23  Color                    1276272 non-null  object  车辆颜色
    #  24  Violation Type           1292399 non-null  object  车辆类型
    #  25  Charge                   1292399 non-null  object  罚款代码
    #  26  Article                  1227230 non-null  object  触犯条例的级别(国家级或者州级) 如： Transportation Article
    #  27  Contributed To Accident  1292399 non-null  object  这次违章是否是导致事故发生的原因
    #  28  Race                     1292399 non-null  object  司机种族
    #  29  Gender                   1292399 non-null  object  司机性别
    #  30  Driver City              1292182 non-null  object  司机家庭住址的城市
    #  31  Driver State             1292388 non-null  object  司机家庭住址的州
    #  32  DL State                 1291470 non-null  object  颁发驾照的州
    #  33  Arrest Type              1292399 non-null  object  是否被逮捕
    #  34  Geolocation              1197045 non-null  object  位置区域码，如 (39.0839483333333, -77.1534983333333)，其实就是由经纬度组成的tuple
    
    
    # 2. convert numeric/categorical columns
    cat_cols = ['SubAgency', 'Description', 'Location', 'Accident', 'Belts', \
        'Personal Injury', 'Property Damage', 'Fatal', 'Commercial License', 'HAZMAT', 'Commercial Vehicle', 'Alcohol', \
        'Work Zone', 'State', 'VehicleType', 'Make', 'Model', 'Color', 'Violation Type', 'Charge', 'Article', \
        'Contributed To Accident', 'Race', 'Gender', 'Driver City', 'Driver State', 'DL State', 'Arrest Type']
    
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')
    
    # 3. compute class label
    #
    # print(df['Violation Type'].value_counts(dropna=False))
    # df = df.loc[df['B'].isin(['one','three'])])
    # Warning     620103  警告
    # Citation    607150  罚单
    # ESERO        64224   Elec. Safety Equip. Repair Order
    # SERO           922   Safety Equip. Repair Order
    # 其中， Warning是程度最轻的，Citation是违反交通规则开具了罚单，ESERO 和 SERO是交通事故后，设备需要维修
    # 把 Warning 当作负样本，其余的当作正样本
    # Charge 和 Arrest Type是相关列，应该去除
    y_col = 'Violation_Type'
    df[y_col] = df['Violation Type'].transform(lambda x: 0 if x=='Warning' else 1)
    df.drop(columns=['Violation Type','Charge', 'Arrest Type'], inplace=True) 
    
    # 4. simple feature extraction
    # 抽取违章事故发生的年份
    # df['Date Of Stop'] = pd.to_datetime(df['Date Of Stop'],format=r'%m/%d/%Y')
    df['Datetime_Of_Stop'] = pd.to_datetime(df['Date Of Stop']+df['Time Of Stop'], format=r'%m/%d/%Y%H:%M:%S')
    df['Year_Of_Stop'] = df['Datetime_Of_Stop'].dt.year  # 35  Year of Stop    1292399 non-null  int64
    df['Month_Of_Stop'] = df['Datetime_Of_Stop'].dt.month
    df['Day_Of_Stop'] = df['Datetime_Of_Stop'].dt.day
    df['Day_Of_Week_Of_Stop'] = df['Datetime_Of_Stop'].dt.dayofweek   
    df['Hour_Of_Stop']= df['Datetime_Of_Stop'].dt.hour 
    df.drop(columns=['Time Of Stop','Date Of Stop','Datetime_Of_Stop'], inplace=True)     
    # print(f"{df['Year_Of_Stop'].min() = }")  # 2012
    # print(f"{df['Year_Of_Stop'].max() = }")  # 2018
    
    # Year （车辆出产日期） 这一列居然有 312 个取值，数据是 2012-2018年的，却有车的出产日期大于2018，
    # print(df['Year'].value_counts(dropna=False))
    # 2006.0    80080
    # 2007.0    79299
    # 2005.0    78297
    # 2004.0    76986
    # 2003.0    73713
     
    # 4313.0        1
    # 4146.0        1
    # 4122.0        1
    # 4112.0        1
    # 1959.0        1
    # Name: Year, Length: 312, dtype: int64
    
    # 只保留生产日期为2012-30=1982到2018年的数据
    df = df[df['Year'].between(1982, 2018)]
    
    # 删掉那些车的出产日期大于出事故的日期的数据，即筛选那些发生事故的时间大于车制造的数据
    df = df.query('Year_Of_Stop > Year').reset_index(drop=True)
    
    
    
    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns
        # 分类任务是预测第24个变量，违章类型，也是2018年Similarity encoding for learning with dirty categorical variables这篇文章的预测任务
        # - Agency 是 unique value，删掉
        # - Geolocation 与 Latitude 和 Longitude 重复，删掉 Geolocation
        # - Location,Description,Model,Driver City是未完全标准化的文本，不在本研究考虑范围
        # ['MASON ST. AT GEORGIA AVE', 'MASON ST @ GRANDVIEW AVE', 'MASON DR/SOUTHLAWN LANE']
        # ['. POTOMAC', '00', '0LNEY', '0XON HILL']
        # ['TOYOTA', 'TOYT', '`TOYOTA']
        
        df.drop(columns=['Agency','Geolocation','Location','Description', 'Model','Driver City'], inplace=True) 
    
        # b. auto identified useless columns
        useless_cols_dict = util.find_useless_colum(df)
        df = util.drop_useless(df, useless_cols_dict, verbose=verbose)
    
    
    # 6. sampling by class
    if num_sample is not None:
        df = util.sampling_by_class(df, y_col, num_sample=num_sample)

        # remove categorical cols with too few samples_per_cat
        #
        # # df statistic
        #         	cardinality	   samples_per_cat	importance_rank	importance_rank_cat
        # SubAgency	      7	              5714.285714	    11	             8
        # Description   1821	           21.96595277	    1	             1
        # Location	    25213	           1.586483163	    2	             2
        # Latitude			                                5	
        # Longitude			                                6	
        # State	        60	               666.6666667	    15	             12
        # VehicleType	 24	               1666.666667	    16	             13    
        # Year		                                        9	
        # Make	        627	               63.79585327	    8	              6
        # Model	        2933	           13.6379134	    4	              4
        # Color	        26	             1538.461538	    10	              7
        # Charge	    525	               76.19047619	     3	              3
        # Race	        6	               6666.666667	    12	              9
        # Gender	    3	               13333.33333	    17	              14
        # Driver City	1375	           29.09090909	    7	              5
        # DL State	    63	               634.9206349	    14	              11
        # Arrest Type	18	               2222.222222	    13	              10
        #    
        # 以上是 small sample 的数据统计，删掉 samples_per_cat 过小的变量 < 30，删掉 Location, Driver City, Model，Description
        # 高基变量选择 Charge，即 罚款代码
        # df.drop(columns=['Description','Location','Model','Driver City'], inplace=True) 
        # print(df['Charge'].value_counts(dropna=False))  # 只有 525了  # avoid zero count of some level
        # print(df.info())
    
    return df, y_col


if __name__ == "__main__":
    df,y_col = load(verbose=True)
    df.info()