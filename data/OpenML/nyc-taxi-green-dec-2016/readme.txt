nyc-taxi-green-dec-2016
https://www.openml.org/search?type=data&sort=runs&id=42729&status=active

% String datetime information extracted to numeric columns.Trip Record Data provided by the New York City Taxi and Limousine Commission (TLC) [http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml]. 
The dataset includes TLC trips of the green line in December 2016. Data was downloaded on 03.11.2018. For a description of all variables in the dataset checkout the TLC homepage [http://www.nyc.gov/html/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf]. The variable 'tip_amount' was chosen as target variable. 
The variable 'total_amount' is ignored by default, otherwise the target could be predicted deterministically. The date variables 'lpep_pickup_datetime' and 'lpep_dropoff_datetime' (ignored by default) could be used to compute additional time features. In this version, we chose only trips with 'payment_type' == 1 (credit card), as tips are not included for most other payment types. 
We also removed the variables 'trip_distance' and 'fare_amount' to increase the importance of the categorical features 'PULocationID' and 'DOLocationID'.
@RELATION nyc-taxi-green-dec-2016

@ATTRIBUTE VendorID {1, 2}
@ATTRIBUTE store_and_fwd_flag {N, Y}
@ATTRIBUTE RatecodeID {1, 2, 3, 4, 5}
@ATTRIBUTE PULocationID {1, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 86, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 101, 102, 106, 107, 108, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 159, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 200, 202, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 225, 226, 227, 228, 229, 230, 231, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 262, 263, 264, 265}
@ATTRIBUTE DOLocationID {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265}
@ATTRIBUTE passenger_count REAL
@ATTRIBUTE extra {0, 0.22, 0.5, 1, 4.5}
@ATTRIBUTE mta_tax {-0.5, 0, 0.5}
@ATTRIBUTE tip_amount REAL
@ATTRIBUTE tolls_amount REAL
@ATTRIBUTE improvement_surcharge {-0.3, 0, 0.3}
@ATTRIBUTE total_amount REAL
@ATTRIBUTE trip_type {1, 2}
@ATTRIBUTE lpep_pickup_datetime_day INTEGER
@ATTRIBUTE lpep_pickup_datetime_hour INTEGER
@ATTRIBUTE lpep_pickup_datetime_minute INTEGER
@ATTRIBUTE lpep_dropoff_datetime_day INTEGER
@ATTRIBUTE lpep_dropoff_datetime_hour INTEGER
@ATTRIBUTE lpep_dropoff_datetime_minute INTEGER
