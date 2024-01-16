particulate-matter-ukair-2017
https://www.openml.org/search?type=data&sort=runs&id=42207&status=active

Hourly particulate matter air polution data of Great Britain for the year 2017, provided by Ricardo Energy and Environment on behalf of the UK Department for Environment, Food and Rural Affairs (DEFRA) and the Devolved Administrations on [https://uk-air.defra.gov.uk/]. 
The data was scraped from the UK AIR homepage via the R-package 'rdefra' [Vitolo, C., Russell, A., & Tucker, A. (2016, August). Rdefra: interact with the UK AIR pollution database from DEFRA. The Journal of Open Source Software, 1(4). doi:10.21105/joss.00051] on 09.11.2018. 
The data was published by DEFRA under the Open Government Licence (OGL) [http://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/]. 
For a description of all variables, checkout the UK AIR homepage. The variable 'PM.sub.10..sub..particulate.matter..Hourly.measured.' was chosen as the target. The dataset also contains another measure of particulate matter 'PM.sub.2.5..sub..particulate.matter..Hourly.measured.' (ignored by default) which could be used as the target instead. 
The string variable 'datetime' (ignored by default) could be used to construct additional date/time features. In this version of the dataset, the features 'Longitude' and 'Latitude' were removed to increase the importance of the categorical features 'Zone' and 'Site.Name'.

@relation R_data_frame

@attribute datetime string
@attribute Hour numeric
@attribute Month {1,2,3,4,5,6,7,8,9,10,11,12}
@attribute DayofWeek {1,2,3,4,5,6,7}
@attribute Site.Name {Aberdeen,'Auchencorth Moss','Barnstaple A39','Belfast Centre','Birmingham A4540 Roadside','Birmingham Tyburn','Bristol St Paul\'s','Camden Kerbside','Cardiff Centre','Carlisle Roadside','Chatham Roadside','Chepstow A48','Chesterfield Loundsley Green','Chesterfield Roadside','Chilbolton Observatory','Derry Rosemount','Edinburgh St Leonards','Glasgow High Street','Glasgow Townhead',Grangemouth,'Greenock A8 Roadside','Leamington Spa','Leamington Spa Rugby Road','Leeds Centre','Leeds Headingley Kerbside','Liverpool Speke','London Bloomsbury','London Harlington','London Marylebone Road','London N. Kensington',Middlesbrough,'Newcastle Centre',Newport,'Norwich Lakenfields','Nottingham Centre','Oxford St Ebbes','Plymouth Centre','Port Talbot Margam',Portsmouth,'Reading New Town','Rochester Stoke','Salford Eccles','Saltash Callington Road','Sandy Roadside','Sheffield Devonshire Green','Southampton Centre','Stanford-le-Hope Roadside','Stockton-on-Tees Eaglescliffe','Storrington Roadside','Swansea Roadside',Warrington,'York Bootham','York Fishergate'}
@attribute Environment.Type {'Background Rural','Background Urban','Industrial Urban','Traffic Urban'}
@attribute Zone {'Belfast Metropolitan Urban Area','Bristol Urban Area','Cardiff Urban Area','Central Scotland','East Midlands',Eastern,'Edinburgh Urban Area','Glasgow Urban Area','Greater London Urban Area','Greater Manchester Urban Area','Liverpool Urban Area','North East','North East Scotland','North West & Merseyside','Northern Ireland','Nottingham Urban Area','Portsmouth Urban Area','Reading/Wokingham Urban Area','Sheffield Urban Area','South East','South Wales','South West','Southampton Urban Area','Swansea Urban Area','Teesside Urban Area',Tyneside,'West Midlands','West Midlands Urban Area','West Yorkshire Urban Area','Yorkshire & Humberside'}
@attribute Altitude..m. numeric
@attribute PM.sub.10..sub..particulate.matter..Hourly.measured. numeric
@attribute PM.sub.2.5..sub..particulate.matter..Hourly.measured. numeric