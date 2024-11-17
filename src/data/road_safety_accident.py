# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 14:50:02 2022
[RoadSafety的官方文档](https://data.dft.gov.uk/road-accidents-safety-data/Road-Safety-Open-Dataset-Data-Guide.xlsx)
数据来源：https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data
下载的是2019年的数据
- 任务是根据一些道路环境因素预测是否会发生严重的事故，只用 accident 这张表即可
- Vehicle 和 Casuality 是事故所涉及的车辆以及受害者信息，逻辑是： 一个事故下，有多辆车参与，可能会涉及多个或者无受害者
- Casuality 的信息不能放进去解释变量去预测是否会发生严重的结果，它们本身就是对事故所造成的结果的统计
@author: YingFu
"""

import pandas as pd
import numpy as np
import os
from zipfile import ZipFile
if __name__ == "__main__":
    import util # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用 

def load(data_dir='../../data', drop_useless=True, num_sample=None, verbose=False): 
    
    # 1. read/uncompress data
    # 文件中的每一行代表唯一的交通事故（由Accident_Index列标识），具有与事故相关的各种属性作为列
    with ZipFile(os.path.join(data_dir, 'data.gov.uk/road_safety/DfTRoadSafety_Accidents_2019.zip')) as zipfile:
        with zipfile.open('Road Safety Data - Accidents 2019.csv') as f:
            df = pd.read_csv(f,low_memory=False)

    # 官方文档中，-1 表示分类变量是 Unknown, Data missing or out of range, Undefined
    # 将 -1 全部替换成 NaN
    df = df.replace(-1, np.NaN)
            
    # print(df.info())
    # RangeIndex: 117536 entries, 0 to 117535
    # Data columns (total 32 columns):
    #  #   Column                                       Non-Null Count   Dtype  
    # ---  ------                                       --------------   -----  
    #  0   Accident_Index                               117536 non-null  object   unique
    #  1   Location_Easting_OSGR                        117508 non-null  float64  
    #  2   Location_Northing_OSGR                       117508 non-null  float64
    #  3   Longitude                                    117508 non-null  float64
    #  4   Latitude                                     117508 non-null  float64
    #  5   Police_Force                                 117536 non-null  int64  
    #  6   Accident_Severity                            117536 non-null  int64  
    #  7   Number_of_Vehicles                           117536 non-null  int64  
    #  8   Number_of_Casualties                         117536 non-null  int64  
    #  9   Date                                         117536 non-null  object  (DD/MM/YYYY)
    #  10  Day_of_Week                                  117536 non-null  int64   (1-7 分别表示星期几)
    #  11  Time                                         117473 non-null  object  (HH:MM)
    #  12  Local_Authority_(District)                   117536 non-null  int64  
    #  13  Local_Authority_(Highway)                    117536 non-null  object 
    #  14  1st_Road_Class                               117536 non-null  int64  
    #  15  1st_Road_Number                              117536 non-null  int64  
    #  16  Road_Type                                    117536 non-null  int64  
    #  17  Speed_limit                                  117536 non-null  int64  
    #  18  Junction_Detail                              117536 non-null  int64  
    #  19  Junction_Control                             117536 non-null  int64  
    #  20  2nd_Road_Class                               117536 non-null  int64  
    #  21  2nd_Road_Number                              117536 non-null  int64  
    #  22  Pedestrian_Crossing-Human_Control            117536 non-null  int64  
    #  23  Pedestrian_Crossing-Physical_Facilities      117536 non-null  int64  
    #  24  Light_Conditions                             117536 non-null  int64  
    #  25  Weather_Conditions                           117536 non-null  int64  
    #  26  Road_Surface_Conditions                      117536 non-null  int64  
    #  27  Special_Conditions_at_Site                   117536 non-null  int64  
    #  28  Carriageway_Hazards                          117536 non-null  int64  
    #  29  Urban_or_Rural_Area                          117536 non-null  int64  
    #  30  Did_Police_Officer_Attend_Scene_of_Accident  117536 non-null  int64  
    #  31  LSOA_of_Accident_Location                    111822 non-null  object 
            
    
    
    # 2. convert numeric/categorical columns
    # cols_num_accidents = ['Location_Easting_OSGR','Location_Northing_OSGR','Longitude',
    #                       'Latitude','Speed_limit']
    
    cols_cat_accidents = ['Police_Force','Day_of_Week','Accident_Severity',
                          'Local_Authority_(District)', 'Local_Authority_(Highway)',
                          '1st_Road_Class','Road_Type','Junction_Detail',
                          'Junction_Control','2nd_Road_Class',
                          'Pedestrian_Crossing-Human_Control','Pedestrian_Crossing-Physical_Facilities',
                          'Light_Conditions','Weather_Conditions','Road_Surface_Conditions', 
                          'Special_Conditions_at_Site', 'Carriageway_Hazards', 
                          'Urban_or_Rural_Area', 'Did_Police_Officer_Attend_Scene_of_Accident',
                          'LSOA_of_Accident_Location']
    
    for col in cols_cat_accidents:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')

    
    # # 官方文档中，-1 表示分类变量是 Unknown, Data missing or out of range, Undefined
    # # 将 -1 全部替换成 NaN
    # df = df.replace(-1, np.NaN)


    # 3. simple feature extraction
    # Feature extraction
    # Date 这个属性 extract month    
    df['Date'] = pd.to_datetime(df['Date'], format=r'%d/%m/%Y')
    df['Date_month'] = df['Date'].dt.month 
    df.drop(columns='Date', inplace=True)

    # Time 这个属性：
    # 1. get part of the day： morning, afternoon, evening, night
    # 2. get the hour,不同的小时，事故发生可能不一样
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
    df['Time_hour'] = df['Time'].dt.hour 
    df.drop(columns='Time', inplace=True)
    
    # 4. compute class label
    y_col = 'Accident_Severity'
    # y为'Accident_Severity'，是以一个accident为一条完整的记录
    # 1	Fatal    1658
    # 2	Serious  23422
    # 3	Slight   92456
    # Name: Accident_Severity, dtype: int64
    df[y_col] = df[y_col].transform(lambda x: 0 if x==3 else 1)# 把 1，2作为正样本 1，把3作为负样本0
    
    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns
        # '1st_Road_Number' 水平太多，且与 1st_Road_Class 重复
        # '2nd_Road_Number' 水平太多，且与 2nd_Road_Class 重复
        df.drop(columns=['1st_Road_Number','2nd_Road_Number'], inplace=True) 

        useless_cols_dict = util.find_useless_colum(df, min_rows_per_value=10)
        df = util.drop_useless(df, useless_cols_dict, verbose=verbose)


    # 6. sampling by class
    if num_sample is not None:
        df = util.sampling_by_class(df, y_col, num_sample)

        # remove categorical cols with too few samples_per_cat

    # print(df.info())
    #  #   Column                                       Non-Null Count   Dtype   
    # ---  ------                                       --------------   -----   
    #  0   Location_Easting_OSGR                        117508 non-null  float64 
    #  1   Location_Northing_OSGR                       117508 non-null  float64 
    #  2   Longitude                                    117508 non-null  float64 
    #  3   Latitude                                     117508 non-null  float64 
    #  4   Police_Force                                 117536 non-null  category
    #  5   Accident_Severity                            117536 non-null  int64   
    #  6   Number_of_Vehicles                           117536 non-null  int64   
    #  7   Number_of_Casualties                         117536 non-null  int64   
    #  8   Day_of_Week                                  117536 non-null  category
    #  9   Local_Authority_(District)                   117536 non-null  category
    #  10  Local_Authority_(Highway)                    117536 non-null  category
    #  11  1st_Road_Class                               117536 non-null  category
    #  12  Road_Type                                    117536 non-null  category
    #  13  Speed_limit                                  117456 non-null  float64 
    #  14  Junction_Detail                              116139 non-null  category
    #  15  Junction_Control                             65160 non-null   category
    #  16  2nd_Road_Class                               68430 non-null   category
    #  17  Pedestrian_Crossing-Physical_Facilities      114294 non-null  category
    #  18  Light_Conditions                             117535 non-null  category
    #  19  Weather_Conditions                           117536 non-null  category
    #  20  Road_Surface_Conditions                      116187 non-null  category
    #  21  Urban_or_Rural_Area                          117536 non-null  category
    #  22  Did_Police_Officer_Attend_Scene_of_Accident  117536 non-null  category
    #  23  Date_month                                   117536 non-null  int64   
    #  24  Time_hour                                    117473 non-null  float64 
    # dtypes: category(15), float64(6), int64(4)  
  
    return df, y_col

if __name__ == "__main__":
    df,y_col = load(verbose=False)    
    df.info()