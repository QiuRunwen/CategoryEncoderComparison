"""
This module contains functions to load datasets which have been cleaned.

@Author: Runwen Qiu
"""

from .bike_sharing import load as load_bike_sharing
from .car_evaluation import load as load_car_evaluation
from .churn import load as load_churn
from .diamonds import load as load_diamonds
from .employee_access import load as load_employee_access
from .employee_salaries import load as load_employee_salaries
from .h1b_visa import load as load_h1b_visa
from .hiv import load as load_hiv
from .housing_price import load as load_housing_price
from .kaggle_cat_in_dat import load as load_kaggle_cat_in_dat
from .kdd_cup09 import load as load_kddcup09
from .kick import load as load_kick
from .lacity_crime import load as load_lacity_crime
from .lending_club import load as load_lending_club
from .misc_colleges import load as load_colleges
from .mushroom import load as load_mushroom
# from .nassCDS import load as load_nassCDS
from .nursery import load as load_nursery
from .nyc_taxi import load as load_nyc_taxi
from .online_retail import load as load_online_retail
from .road_safety_accident import load as load_road_safety_accident
from .student_performance import load as load_student_performance
from .tianchi_auto_loan_default_risk import \
    load as load_tianchi_auto_loan_default_risk
from .traffic_violation import load as load_traffic_violation
from .uci_adult import load as load_adult
from .ukair import load as load_ukair
from .autism import load as load_autism
from .mammographic import load as load_mammographic
from .obesity import load as load_obesity
from .german_credit import load as load_german_credit
from .trip_advisor import load as load_trip_advisor
from .wholesale import load as load_wholesale
from .cholesterol import load as load_cholesterol
from .cpmp2015 import load as load_cpmp2015
from .avocado import load as load_avocado
from .socmob import load as load_socmob
from .moneyball import load as load_moneyball
from .chscase_foot import load as load_chscase_foot
from .cps1988 import load as load_cps1988

__all__ = [
    "load_bike_sharing",
    "load_car_evaluation",
    "load_churn",
    "load_diamonds",
    "load_employee_access",
    "load_employee_salaries",
    "load_h1b_visa",
    "load_hiv",
    "load_housing_price",
    "load_kaggle_cat_in_dat",
    "load_kddcup09",
    "load_kick",
    "load_lacity_crime",
    "load_lending_club",
    "load_colleges",
    "load_mushroom",
    # "load_nassCDS",
    "load_nursery",
    "load_nyc_taxi",
    "load_online_retail",
    "load_road_safety_accident",
    "load_student_performance",
    "load_tianchi_auto_loan_default_risk",
    "load_traffic_violation",
    "load_adult",
    "load_ukair",
    "load_autism",
    "load_mammographic",
    "load_obesity",
    "load_german_credit",
    "load_trip_advisor",
    "load_wholesale",
    "load_cholesterol",
    "load_cpmp2015",
    "load_avocado",
    "load_socmob",
    "load_moneyball",
    "load_chscase_foot",
    "load_cps1988"
]
