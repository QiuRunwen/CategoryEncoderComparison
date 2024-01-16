# Us H1-B Visa Application Data (2011-2016)

- h1b_kaggle.csv    2020-11-24    469.45MB
- 3002458行，10列

URL: https://www.kaggle.com/nsharan/h-1b-visa/version/2
URL: https://tianchi.aliyun.com/dataset/dataDetail?dataId=83988#1


H-1B visas are a category of employment-based, non-immigrant visas for temporary foreign workers in the United States. For a foreign national to apply for H1-B visa, a US employer must offer them a job and submit a petition for a H-1B visa to the US immigration department. This is also the most common visa status applied for and held by international students once they complete college or higher education and begin working in a full-time position.

The following articles contain more information about the H-1B visa process:

What is H1B LCA ? Why file it ? Salary, Processing times – DOL
H1B Application Process: Step by Step Guide
Content
This dataset contains five year's worth of H-1B petition data, with approximately 3 million records overall. The columns in the dataset include case status, employer name, worksite coordinates, job title, prevailing wage, occupation code, and year filed.

For more information on individual columns, refer to the column metadata. A detailed description of the underlying raw dataset is available in an official data dictionary.

Acknowledgements
The Office of Foreign Labor Certification (OFLC) generates program data, including data about H1-B visas. The disclosure data updated annually and is available online.

The raw data available is messy and not immediately suitable analysis. A set of data transformations were performed making the data more accessible for quick exploration. To learn more, refer to this blog post and to the complimentary R Notebook.

Inspiration
Is the number of petitions with Data Engineer job title increasing over time?
Which part of the US has the most Hardware Engineer jobs?
Which industry has the most number of Data Scientist positions?
Which employers file the most petitions each year?




|                    |                                                              |                                                              | cardinality |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------- |
| CASE_STATUS        | 签证状态                                                     | CERTIFIED	2615623<br/>CERTIFIED-WITHDRAWN	202659<br/>DENIED	94346<br/>WITHDRAWN	89799<br/>PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED	15<br/>REJECTED	2<br/>INVALIDATED	1 | 7           |
| EMPLOYER_NAME      | 雇主                                                         | Top3是：INFOSYS LIMITED，TATA CONSULTANCY SERVICES LIMITED，WIPRO LIMITED | 236013      |
| SOC_NAME           | 美国标准职业分类系统（Standard Occupational Classification 简称SOC） | Top3是：‘Computer Systems Analysts‘，’Computer Programmers‘，’SOFTWARE DEVELOPERS, APPLICATIONS‘ | 2132        |
| JOB_TITLE          |                                                              | Top3是：PROGRAMMER ANALYST，SOFTWARE ENGINEER，COMPUTER PROGRAMMER | 287549      |
| FULL_TIME_POSITION |                                                              | Y	2576111<br/>N	426332                                 | 2           |
| PREVAILING_WAGE    | 可支配工资                                                   |                                                              |             |
| YEAR               |                                                              | 2016.0	647803<br/>2015.0	618727<br/>2014.0	519427<br/>2013.0	442114<br/>2012.0	415607<br/>2011.0	358767 |             |
| WORKSITE           | 工作地点                                                     |                                                              | 18622       |
| lon                |                                                              |                                                              |             |
| lat                |                                                              |                                                              |             |



