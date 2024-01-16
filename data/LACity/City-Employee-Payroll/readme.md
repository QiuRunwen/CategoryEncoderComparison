> Created on 2019-11-29
>
> @author: WXT (847718009@qq.com)

------

City_Employee_Payroll.csv

#### 来源：

- ⭐来源： https://controllerdata.lacity.org/Payroll/City-Employee-Payroll/pazn-qyym 



#### Description

-  Payroll information for all Los Angeles City Employees including the City's three proprietary departments: Water and Power, Airports and Harbor. Data is updated on a quarterly basis by the Los Angeles City Controller's Office. Payroll information for employees of the Department of Water and Power is provided by the Department. 
-  Data Last Updated :  June 6, 2019 
- size: 122M 



#### Data dictionary

371456 rows, 35 cols

| COLUMN NAME                  | Description                                                  | Type        | Unique |
| ---------------------------- | ------------------------------------------------------------ | ----------- | ------ |
| Row ID                       | Unique Identifier for each row                               | Plain Text  |        |
| Year                         | Calendar Year                                                | Number      |        |
| Department Title             | Title of City Department                                     | Plain Text  | 91     |
| Payroll Department           | Department Number in City Payroll System                     | Plain Text  | 216    |
| Record Number                |                                                              | Plain Text  |        |
| Job Class Title              |                                                              | Plain Text  | 1747   |
| Employment Type              | <font size=2>Employment Type - Full Time, Part Time, or Per Event</font> | Plain Text  | 3      |
| Hourly or Event Rate         | <font size=2>Hourly Earnings Rate or Per Event Rate based on Projected Annual Salary</font> | Number      |        |
| Projected Annual Salary      | <font size=2>Budgeted pay amount. Used for pension contribution calculations</font> | Number      |        |
| Q1 Payments                  | <font size=2>Payments for the first quarter of the year from January 1 to March 31</font> | Number      |        |
| Q2 Payments                  | <font size=2>Payments for the second quarter of the year from April 1 to June 30</font> | Number      |        |
| Q3 Payments                  | <font size=2>Payments for the third quarter of the year from July 1 to September 30</font> | Number      |        |
| Q4 Payments                  | <font size=2>Payments for the fourth quarter of the year from October 1 to December 31</font> | Number      |        |
| Payments Over Base Pay       | <font size=2>Payments in excess of Base Pay which may include bonuses and other payouts</font> | Number      |        |
| % Over Base Pay              | <font size=2>Percentage of payment in excess of Base Pay which may include bonuses and other payouts</font> | Number      |        |
| Total Payments               | Total earnings for the year                                  | Number      |        |
| Base Pay                     | Base compensation for hours worked                           | Number      |        |
| Permanent Bonus Pay          | <font size=2>Payments attributable to permanent bonuses; typically pensionable</font> | Number      |        |
| Longevity Bonus Pay          | <font size=2>Payments attributable to years of service; typically pensionable</font> | Number      |        |
| Temporary Bonus Pay          | <font size=2>Payments attributable to temporary bonuses; typically not pensionable</font> | Number      |        |
| Lump Sum Pay                 | <font size=2>Lump sum payouts for special purposes - retirement payouts, back pay, etc.; typically not pensionable</font> | Number      |        |
| Overtime Pay                 | <font size=2>Payments attributable to hours worked beyond regular work schedule</font> | Number      |        |
| Other Pay & Adjustments      | <font size=2>Payments based on other pay codes or adjustments that do not fall into another category</font> | Number      |        |
| Other Pay (Payroll Explorer) | <font size=2>Other Pay includes bonuses, adjustments, and lump sum payouts. Examples of bonuses include Permanent, Longevity, and Temporary Bonuses. Lump Sum Pay includes significant one-time payouts due to retirement, lawsuit settlements, or other adjustments</font> | Number      |        |
| MOU                          | Memorandum of Understanding                                  | Plain Text  | 57     |
| MOU Title                    | Title of Memorandum of Understanding                         | Plain Text  | 96     |
| FMS Department               | <font size=2>Department number in City Financial Management System</font> | Plain Text  | 48     |
| Job Class                    |                                                              | Plain Text  | 1180   |
| Pay Grade                    | Pay Grade for the Job Class                                  | Plain Text  |        |
| Average Health Cost          | <font size=2>Average cost to the City to provide health care to the employee</font> | Number      |        |
| Average Dental Cost          | <font size=2>Average cost to the City to provide dental care to the employee</font> | Number      |        |
| Average Basic Life           | <font size=2>Average cost to the City to provide basic life insurance to the employee</font> | Number      |        |
| Average Benefit Cost         | <font size=2>The total average City contribution for the employee's health care, dental care and life insurance</font> | Number      |        |
| Benefits Plan                |                                                              | Plain Text  |        |
| Job Class Link               | <font size=2>Click this hyperlink to view the job class description</font> | Website URL |        |