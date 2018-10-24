import pandas as pd

from pyEDA.data_profile import ExploratoraryDataAnalysis as eda
'''
'''
#########################
'''
data = pd.read_csv("src\\df.csv")
print("Data Read")
# checking the data availability month wise for date-time column
eda.data_profile(dataframe = data,file_name = 'bsg'
                 ,date_time_var = 'week_date_li', target_variable = 'units', col_to_be_excluded = [])
# This plot we can see that for month of 'April 2017' there is no data in 'Calendar_Intro_Dt'
'''
#########################
'''

data = pd.read_csv("src\\final_df.csv")
print("Data Read")
# 2- checking the data availability month wise for date-time column
eda.data_profile(dataframe = data,file_name = 'bsg')
# Here we can see the skewness and warnings in the data, we can also interpret the missing value pattern


'''

'''
#########################

data = pd.read_csv("src\\df.csv")
print("Data Read")
# 3- checking the data availability month wise for date-time column
eda.data_profile(dataframe = data,file_name = 'profile'
                 ,date_time_var = 'businessdate', target_variable = 'quantity', col_to_be_excluded = [])
# Here we can see warnings and in 2018 there is no data after june


'''



#########################
print("Imported data_profile")
data = pd.read_csv("pyEDA\\df.csv")
print("Data Read")
# 3- checking the data availability month wise for date-time column
eda.profile(dataframe = data, date_time_var = ['businessdate', 'businessdate_1'])
# Here we can see warnings and in 2018 there is no data after june




