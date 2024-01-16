> Created on 2019-11-29
>
> @author: WXT (847718009@qq.com)

------

beer_reviews.csv

#### 来源：

- dirty-cat项目（github）： https://github.com/dirty-cat/datasets/blob/master/src/beer_reviews.py   ；  https://github.com/dirty-cat/datasets/blob/master/src/openml_beer_upload.py  
- ⭐原始来源：https://data.world/socialmediadata/beeradvocate
- 下载链接：https://query.data.world/s/vmplvzsgmb2gdcomuhwktnzt2laoiy



#### Description

- This dataset consists of beer reviews from Beeradvocate. The data span a period of more than 10 years, including all ~1.5 million reviews up to November 2011. Each review includes ratings in  terms of five "aspects": appearance, aroma, palate, taste, and overall impression. Reviews  include product and user information, followed by each of these five ratings, and a plaintext  review. We also have reviews from ratebeer. 
- default_target_attribute:  Beer_Style 
- size: 171M 



#### Data dictionary

1586614rows, 13cols

| COLUMN NAME        | TYPE    | DISTINCT | EMPTY |
| ------------------ | ------- | -------- | ----- |
| brewery_id         | integer | >1000    | 0     |
| brewery_name       | string  | >1000    | 0     |
| review_time        | integer | >1000    | 0     |
| review_overall     | decimal | 10       | 0     |
| review_aroma       | decimal | 9        | 0     |
| review_appearance  | decimal | 10       | 0     |
| review_profilename | string  | 33387    | 0     |
| beer_style         | string  | 104      | 0     |
| review_palate      | decimal | 9        | 0     |
| review_taste       | decimal | 9        | 0     |
| **beer_name**      | string  | 56857    | 0     |
| beer_abv           | decimal | 530      | 0     |
| beer_beerid        | integer | >1000    | 0     |

