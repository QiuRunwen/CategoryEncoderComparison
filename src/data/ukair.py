"""
particulate-matter-ukair-2017

Hourly particulate matter air polution data of Great Britain for the year 2017,
provided by Ricardo Energy and Environment on behalf of the UK Department for Environment,
Food and Rural Affairs (DEFRA) and the Devolved Administrations on [https://uk-air.defra.gov.uk/].

https://www.openml.org/d/42207

Author: QiuRunwen
"""

import pandas as pd
import os

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load(data_dir="../../data", drop_useless=True, num_sample=None, verbose=False):
    # 1. read/uncompress data
    file_path = os.path.join(
        data_dir,
        "OpenML/particulate-matter-ukair-2017",
        "particulate-matter-ukair-2017.csv",
    )
    df = pd.read_csv(
        file_path, parse_dates=["datetime"], date_format="%Y/%m/%d %H:%M:%S"
    )

    # df.info()
    # 394299*10
    # @attribute datetime string
    # @attribute Hour numeric
    # @attribute Month {1,2,3,4,5,6,7,8,9,10,11,12}
    # @attribute DayofWeek {1,2,3,4,5,6,7}
    # @attribute Site.Name {Aberdeen,'Auchencorth Moss','Barnstaple A39','Belfast Centre','Birmingham A4540 Roadside'
    # ,'Birmingham Tyburn','Bristol St Paul\'s','Camden Kerbside','Cardiff Centre','Carlisle Roadside','Chatham Roadside'
    # ,'Chepstow A48','Chesterfield Loundsley Green','Chesterfield Roadside','Chilbolton Observatory','Derry Rosemount'
    # ,'Edinburgh St Leonards','Glasgow High Street','Glasgow Townhead',Grangemouth,'Greenock A8 Roadside','Leamington Spa'
    # ,'Leamington Spa Rugby Road','Leeds Centre','Leeds Headingley Kerbside','Liverpool Speke','London Bloomsbury'
    # ,'London Harlington','London Marylebone Road','London N. Kensington',Middlesbrough,'Newcastle Centre'
    # ,Newport,'Norwich Lakenfields','Nottingham Centre','Oxford St Ebbes','Plymouth Centre','Port Talbot Margam'
    # ,Portsmouth,'Reading New Town','Rochester Stoke','Salford Eccles','Saltash Callington Road','Sandy Roadside'
    # ,'Sheffield Devonshire Green','Southampton Centre','Stanford-le-Hope Roadside','Stockton-on-Tees Eaglescliffe'
    # ,'Storrington Roadside','Swansea Roadside',Warrington,'York Bootham','York Fishergate'}
    # @attribute Environment.Type {'Background Rural','Background Urban','Industrial Urban','Traffic Urban'}
    # @attribute Zone {'Belfast Metropolitan Urban Area','Bristol Urban Area','Cardiff Urban Area'
    # ,'Central Scotland','East Midlands',Eastern,'Edinburgh Urban Area','Glasgow Urban Area','Greater London Urban Area'
    # ,'Greater Manchester Urban Area','Liverpool Urban Area','North East','North East Scotland','North West & Merseyside'
    # ,'Northern Ireland','Nottingham Urban Area','Portsmouth Urban Area','Reading/Wokingham Urban Area'
    # ,'Sheffield Urban Area','South East','South Wales','South West','Southampton Urban Area','Swansea Urban Area'
    # ,'Teesside Urban Area',Tyneside,'West Midlands','West Midlands Urban Area','West Yorkshire Urban Area','Yorkshire & Humberside'}
    # @attribute Altitude..m. numeric
    # @attribute PM.sub.10..sub..particulate.matter..Hourly.measured. numeric
    # @attribute PM.sub.2.5..sub..particulate.matter..Hourly.measured. numeric

    # 2. convert numeric/categorical columns
    cat_cols = ["Site.Name", "Environment.Type", "Zone", "Month", "DayofWeek"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction
    # datetime已经从Hour,Month,DayofWeek提取了feature，而且只有year只有2017.所以不再额外需要处理
    df.drop(columns="datetime", inplace=True)

    # 4. compute target
    # PM2.5是指环境空气中空气动力学直径小于等于 2.5 微米的颗粒物
    # PM2.5比PM10危害更大，并且更加受关注http://www.huizhou.gov.cn/zmhd/zczx/sthj/content/post_4874382.html
    # 选取PM2.5。另外PM2.5是PM10的一部分，去除PM10。
    # 目前名字太长，换个短一点的
    y_col = "PM2.5_hourly"
    df[y_col] = df["PM.sub.2.5..sub..particulate.matter..Hourly.measured."]

    df.drop(
        columns=[
            "PM.sub.10..sub..particulate.matter..Hourly.measured.",
            "PM.sub.2.5..sub..particulate.matter..Hourly.measured.",
        ],
        inplace=True,
    )

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
    df, y_col = load(verbose=True)
    df.info()
