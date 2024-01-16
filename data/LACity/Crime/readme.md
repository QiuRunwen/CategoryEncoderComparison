> Created on 2019-11-29
>
> Last updated： 
>
> @author: WXT (847718009@qq.com)

------

Crime_Data_from_2010_to_Present.csv

#### 来源：

- dirty-cat项目（github）： https://github.com/dirty-cat/datasets/blob/master/src/crime_data.py ；   https://github.com/dirty-cat/datasets/blob/master/src/openml_crime_upload.py 
- ⭐原始来源（含下载）： https://data.lacity.org/A-Safe-City/Crime-Data-from-2010-to-Present/63jg-8b9z (翻墙)

#### Description

- This dataset reflects incidents of crime in the City of Los Angeles dating back to 2010. This data is transcribed from original crime reports that are typed on paper and therefore there may be some inaccuracies within the data. Some location fields with missing data are noted as (0°, 0°). Address fields are only provided to the nearest hundred block in order to maintain privacy. This data is as accurate as the data in the database. Please note questions or concerns in the comments. 

  （本数据集包含了洛杉矶2010年起的犯罪数据集。这些数据是从纸质的原始交通报告转录而来，因此数据中可能存在一些不准确之处。一些缺少数据的位置字段标记为（0°，0°））

- Updated  November 27, 2019 

- size: 258M 

- 含有大量的缺失值



#### Data dictionary

1074524 rows, 28 cols

| COLUMN NAME    | Description                                                  | TYPE        | unique | non-null |
| -------------- | ------------------------------------------------------------ | ----------- | ------ | -------- |
| DR_NO          | <font size=2>Division of Records Number（分类编号）: Official file number made up of a 2 digit year, area ID, and 5 digits </font> | Plain Text  |        | 1074524  |
| Date Rptd      | MM/DD/YYYY                                                   | Date & Tim  | 3504   | 1074524  |
| DATE OCC       | MM/DD/YYYY                                                   | Date & Time | 2191   | 1074524  |
| TIME OCC       | In 24 hour military time.                                    | Plain Text  |        | 1074524  |
| AREA           | <font size=2>The LAPD has 21 Community Police Stations referred to as Geographic Areas within the department. These Geographic Areas are sequentially numbered from 1-21. （洛杉矶有21个社区警察局。编号为1-21）</font> | Plain Text  |        | 1074524  |
| AREA NAME      | <font size=2>The 21 Geographic Areas or Patrol Divisions are also given a name designation that references a landmark or the surrounding community that it is responsible for. For example 77th Street Division is located at the intersection of South Broadway and 77th Street, serving neighborhoods in South Los Angeles.（21个警察局的名字）</font> | Plain Text  | 21     | 1074524  |
| Rpt Dist No    | <font size=2>A four-digit code that represents a sub-area within a Geographic Area. All crime records reference the "RD" that it occurred in for statistical comparisons. Find LAPD Reporting Districts on the LA City GeoHub at http://geohub.lacity.org/datasets/c4f83909b81d4786aa8ba8a74a4b4db1_4  （Reporting District：用于生成报告以将数据分组到区域内的地理子区域中的代码。）</font> | Plain Text  | 1266   | 1074524  |
| Part 1-2       |                                                              | Number      |        | 1074524  |
| Crm Cd         | Indicates the crime committed. (Same as Crime Code 1)        | Plain Text  | 136    | 1074524  |
| Crm Cd Desc    | Defines the Crime Code provided.                             | Plain Text  | 136    | 1074524  |
| Mocodes        | <font size=2>Modus Operandi: Activities associated with the suspect in commission of the crime.（操作方式：与犯罪嫌疑人有关的活动。）See attached PDF for list of MO Codes in numerical order. [https://data.lacity.org/api/views/y8tr-7khq/files/3a967fbd-f210-4857-bc52-60230efe256c?download=true&filename=MO%20CODES%20(numerical%20order).pdf](https://data.lacity.org/api/views/y8tr-7khq/files/3a967fbd-f210-4857-bc52-60230efe256c?download=true&filename=MO CODES (numerical order).pdf) </font> | Plain Text  | 233055 | 956809   |
| Vict Age       | Two character numeric                                        | Plain Text  |        | 1074524  |
| Vict Sex       | F - Female M - Male X - Unknown                              | Plain Text  | 5      | 976996   |
| Vict Descent   | <font size=2>Descent Code: A - Other Asian B - Black C - Chinese D - Cambodian F - Filipino G - Guamanian H - Hispanic/Latin/Mexican I - American Indian/Alaskan Native J - Japanese K - Korean L - Laotian O - Other P - Pacific Islander S - Samoan U - Hawaiian V - Vietnamese W - White X - Unknown Z - Asian Indian </font> | Plain Text  | 20     | 976976   |
| Premis Cd      | The type of structure, vehicle, or location where the crime took place. | Plain Text  | 223    | 1074486  |
| Premis Desc    | Defines the Premise Code provided.                           | Plain Text  | 222    | 1074486  |
| Weapon Used Cd | The type of weapon used in the crime.                        | Plain Text  |        | 355327   |
| Weapon Desc    | Defines the Weapon Used Code provided.                       | Plain Text  | 78     | 355326   |
| Status         | Status of the case. (IC is the default)                      | Plain Text  | 8      | 1074522  |
| Status Desc    | Defines the Status Code provided.                            | Plain Text  | 6      | 1074524  |
| Crm Cd 1       | <font size=2>Indicates the crime committed. Crime Code 1 is the primary and most serious one. Crime Code 2, 3, and 4 are respectively less serious offenses. Lower crime class numbers are more serious. </font> | Plain Text  |        | 1074520  |
| Crm Cd 2       | <font size=2>May contain a code for an additional crime, less serious than Crime Code 1.</font> | Plain Text  |        | 66341    |
| Crm Cd 3       | <font size=2>May contain a code for an additional crime, less serious than Crime Code 1.</font> | Plain Text  |        | 1161     |
| Crm Cd 4       | <font size=2>May contain a code for an additional crime, less serious than Crime Code 1.</font> | Plain Text  |        | 41       |
| LOCATION       | <font size=2>Street address of crime incident rounded to the nearest hundred block to maintain anonymity.</font> | Plain Text  | 66136  | 1074524  |
| Cross Street   | <font size=2>Cross Street of rounded Address</font>          | Plain Text  | 9492   | 176913   |
| LAT            | Latitude                                                     | Number      |        | 1074524  |
| LON            | Longtitude                                                   | Number      |        | 1074524  |

`AREA`，`AREA NAME`：洛杉矶警署地理区域，21个

📕**TODO**：`Vict Descent`作案手法：与犯罪嫌疑人有关的活动，还可进一步研究

#### Analysis

----

- 先剔除 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4' 三列，然后剔除缺失值之后，剩下 90354 rows，25 cols

```python
import pandas as pd
df = pd.read_csv('D:\github\CategoricalEncoder\dataset\cirme\Crime_Data_from_2010_to_Present.csv')
# 剔除 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4' （缺失值比例过高）
df = df.drop('Crm Cd 2', axis=1)
df = df.drop('Crm Cd 3', axis=1)
df = df.drop('Crm Cd 4', axis=1)
# 剔除缺失值
df = df.dropna()

df.shape
```

```
Out[30]: (90354, 25)
```



-  https://github.com/dirty-cat/datasets/blob/master/src/crime_data.py  中选用的 cols 有7个：

```python
cols = ['Area Name', 'Victim Sex', 'Victim Descent', 'Premise Description', 'Weapon Description', 'Status Description', 'Crime Code Description']
```

- target variable 为： `Crm Cd 1` （cardinality = 103）**(TODO: 怎么变成二分类？）**

----

****

**target variable ：Status**

| Status | Status Desc            | 猜测含义                       |
| ------ | ---------------------- | ------------------------------ |
| IC     | Invert Cont （默认值） | 还在侦查阶段，还没破案         |
| AA     | Adult Arrest           | 已破案，罪犯是成年人且已逮捕   |
| AO     | Adult Other            | 已破案，罪犯是成年人且...      |
| JA     | Juv Arrest             | 已破案，罪犯是未成年人且已逮捕 |
| JO     | Juv Other              | 已破案，罪犯是未成年人且...    |
| CC     | UNK                    | 未知                           |
| TH     | UNK                    | 未知                           |
| 13     | UNK                    | 未知                           |

- 目标变量划分：剔除CC，TH，13，把 IC 设为 1，把AA，AO，JA，JO设为 0

- 衍生变量：`df['time_interval'] = df['Date Rptd'] - df['DATE OCC']`
  - 时间间隔 = 报告时间 - 犯罪发生时间

- 剔除`'Date Rptd'`，再把 `DATE OCC`分解成年、月、日、星期
- 只选择 2015 年的样本，处理之后：67878 rows x 13 cols
  - 保存在本地：`'D:/github/CategoricalEncoder/dataset/cirme/crime2015.csv'`



---------

#### 参考

- https://www.kesci.com/mw/dataset/5d396b6acf76a6003608fff9/content

> ## **背景描述**
>
> 本数据集包含了洛杉矶2010-2019年的交通事故数据集。这些数据是从纸质的原始交通报告转录而来，因此数据中可能存在一些不准确之处。一些缺少数据的位置字段标记为（0°，0°）。
>
> ## **数据说明**
>
> DR Number：分类编号
> Date Reported：MM / DD / YYYY
> Date Occurred：MM / DD / YYYY
> Time Occurred：24小时
> Area ID：洛杉矶有21个社区警察局。编号为1-21。
> Area Name：21个警察局的名字。
> Reporting District：用于生成报告以将数据分组到区域内的地理子区域中的代码。
> Crime Code：犯罪行为。对于这个数据集，都为997。
> Crime Code Description：定义crime code。
> MO Codes
> Victim Age
> Victim SexF
> Victim Descent：A - Other Asian B - Black C - Chinese D - Cambodian F - Filipino G - Guamanian H - Hispanic/Latin/Mexican I - American Indian/Alaskan Native J - Japanese K - Korean L - Laotian O - Other P - Pacific Islander S - Samoan U - Hawaiian V - Vietnamese W - White X - Unknown Z - Asian Indian
> 等
>
> ## **数据来源**
>
> https://www.kaggle.com/cityofLA/los-angeles-traffic-collision-data