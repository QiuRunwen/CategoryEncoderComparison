参考：https://github.com/ictar/python-doc/blob/master/Science%20and%20Data%20Analysis/%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8Python%E5%92%8CPandas%E5%A4%84%E7%90%86%E5%A4%A7%E9%87%8F%E7%9A%84JSON%E6%95%B0%E6%8D%AE%E9%9B%86.md



~~TODO：去这里找 https://catalog.data.gov/dataset （找不到）~~



## 数据集

我们将着眼于一个包含Maryland州Montgomery郡交通违法行为信息的数据集。你也可以在[这里](https://catalog.data.gov/dataset/traffic-violations-56dda)下载数据。该数据包含有关违规发生的地点信息，车的类型，关于收到违规的人的人口统计，以及其他一些有趣的信息。我们可以用这个数据集回答相当多的问题，其中包括了几个问题：

- 什么类型的汽车是最有可能被超速拦停？
- 一天的哪些时候警察最活跃？
- “超速陷阱”有多常见？或者在地理位置方面，交通罚款单是否相当均匀分布？
- 人们被拦下来最常见的原因是什么？

> ## The dataset
>
> We’ll be looking at a dataset that contains information on traffic violations in Montgomery County, Maryland. You can download the data [here](https://catalog.data.gov/dataset/traffic-violations-56dda). The data contains information about where the violation happened, the type of car, demographics on the person receiving the violation, and some other interesting information. There are quite a few questions we could answer using this dataset, including:
>
> - What types of cars are most likely to be pulled over for speeding?
> - What times of day are police most active?
> - How common are “speed traps”? Or are tickets spread pretty evenly in terms of geography?
> - What are the most common things people are pulled over for?



来源：2018-Similarity encoding for learning with dirty categorical variables.pdf  ——待核实

> Traffic violations. Traffic information from electronic violations issued in the Montgomery County of Maryland. Sample size (random subsample): 100,000. Target variable(multiclass-clf): ‘Violation type’ (4 classes). Selected categorical variable: ‘Description’(card.: 3043). Other explanatory variables: ‘Belts’ (c), ‘Property Damage’ (c), ‘Fatal’ (c),‘Commercial license’ (c), ‘Hazardous materials’ (c), ‘Commercial vehicle’ (c), ‘Alcohol’(c), ‘Work zone’ (c), ‘Year’ (n), ‘Race’ (c), ‘Gender’ (c), ‘Arrest type’ (c).  
>
> https://catalog.data.gov/dataset/traffic-violations-56dda  



> GitHub项目：[dirty_cat](https://github.com/dirty-cat/dirty_cat/blob/master/dirty_cat/datasets/fetching.py)
>
> GitHub项目：[Traffic-Violations-Analysis](https://github.com/meliharici/Traffic-Violations-Analysis)
>
> https://data.world/jrm/traffic-violations
>
> https://github.com/ChristianNHill/The-Mean-Lovers
>
> https://kooplex-edu.elte.hu/report/wfct0p/testinteractivehtml/2019_10_08-17:29:40/Example-interactive.html
>
> https://data.amerigeoss.org/dataset/traffic-violations-56dda



> https://www.kaggle.com/rounak041993/traffic-violations-in-maryland-county  这里可以下载（20210321）⭐⭐

该数据集包含从2012年到2018年的所有交通违规事件。它有约129万+条记录（2021-03-21下载）。

**The data include items, such as:**

Accident : If traffic violation involved an accident.

Agency : Agency issuing the traffic violation. (Example: MCP is Montgomery County Police)

Alcohol : If the traffic violation included an alcohol related

Arrest Type : Type of Arrest (A = Marked, B = Unmarked, etc.)

Article : Article of State Law. (TA = Transportation Article, MR = Maryland Rules)

Belts : If traffic violation involved a seat belt violation.

Charge : Numeric code for the specific charge.

Color : Color of the vehicle.

Commercial License : If driver holds a Commercial Drivers License.

Commercial Vehicle : If the vehicle committing the traffic violation is a commercial vehicle.

Contributed To Accident : If the traffic violation was a contributing factor in an accident.

Date Of Stop : Date of the traffic violation.

Description : Text description of the specific charge.

DL State : State issuing the Driver’s License.

Driver City : City of the driver’s home address.

Driver State : State of the driver’s home address.

Fatal : If traffic violation involved a fatality.

Gender : Gender of the driver (F = Female, M = Male)

Geolocation : Geo-coded location information.

HAZMAT : If the traffic violation involved hazardous materials.

Latitude : Latitude location of the traffic violation.

Location : Location of the violation, usually an address or intersection.

Longitude : Longitude location of the traffic violation.

Make : Manufacturer of the vehicle (Examples: Ford, Chevy, Honda, Toyota, etc.)

Model : Model of the vehicle.

Personal Injury : If traffic violation involved Personal Injury.

Property Damage : If traffic violation involved Property Damage.

Race : Race of the driver. (Example: Asian, Black, White, Other, etc.)

State : State issuing the vehicle registration.

SubAgency : Court code representing the district of assignment of the officer. R15 = 1st district, Rockville B15 = 2nd
district, Bethesda SS15 = 3rd district, Silver Spring WG15 = 4th district, Wheaton G15 = 5th district, Germantown M15 = 6th district, Gaithersburg / Montgomery Village HQ15 = Headquarters and Special Operations

Time Of Stop : Time of the traffic violation.

VehicleType : Type of vehicle (Examples: Automobile, Station Wagon, Heavy Duty Truck, etc.)

Violation Type : Violation type. (Examples: Warning, Citation, SERO)

Work Zone : If the traffic violation was in a work zone.

Year : Year vehicle was made.

***The time period of this data ranges from 2012-2018\***