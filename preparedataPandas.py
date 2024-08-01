import numpy as np
import csv
import pandas as pd 

#Read the pressure.csv file into first a pressure_data_iter
#From the iterator, create the pressure_data array 

with open("pressure.csv",'r') as dest_f:
    pressure_data_iter = csv.reader(dest_f,
                           delimiter = ',',
                           quotechar = '"')
    pressure_data = [pressure_data for pressure_data in pressure_data_iter]

#Convert the python array into a numpy array 

pressure_data_array = np.asarray(pressure_data)

#Print the shape of the numpy array 
print("pressure_data_array shape=",pressure_data_array.shape)
print("pressure_data_array=",pressure_data_array)

#Read pressure2 into pressure2 data array 
with open("pressure2.csv",'r') as dest_f:
    pressure2_data_iter = csv.reader(dest_f,
                           delimiter = ',',
                           quotechar = '"')
    pressure2_data = [pressure2_data for pressure2_data in pressure2_data_iter]
pressure2_data_array = np.asarray(pressure2_data)
print("pressure2_data_array shape=",pressure2_data_array.shape)
print("pressure2_data_array=",pressure2_data_array)

#Read wind into wind data array 
with open("wind.csv",'r') as dest_f:
    wind_data_iter = csv.reader(dest_f,
                           delimiter = ',',
                           quotechar = '"')
    wind_data = [wind_data for wind_data in wind_data_iter]
wind_data_array = np.asarray(wind_data)
print("wind_data_array shape=",wind_data_array.shape)
print("wind_data_array=",wind_data_array)

#Next 3 lines get the lengths of the 3 arrays 
pressure_size = pressure_data_array.shape[0]
pressure2_size = pressure2_data_array.shape[0]
wind_size = wind_data_array.shape[0]

#Array size is the min. of the 3 
array_size = min(pressure_size, pressure2_size,wind_size)
print("array_size=",array_size)

#The next lines get the min. & max. of all the arrays 
minptime = np.min(pressure_data_array[:,2].astype(np.float64)) 
print("minptime=",minptime)
maxptime = np.max(pressure_data_array[:,2].astype(np.float64)) 
print("maxptime=",maxptime)

minp2time = np.min(pressure2_data_array[:,2].astype(np.float64)) 
print("minp2time=",minp2time)
maxp2time = np.max(pressure2_data_array[:,2].astype(np.float64)) 
print("maxp2time=",maxp2time)

minwtime = np.min(wind_data_array[:,2].astype(np.float64))
print("minwtime=",minwtime)
maxwtime = np.max(wind_data_array[:,2].astype(np.float64))
print("maxwtime=",maxwtime)

mintime = max(minptime,minp2time,minwtime)
print("mintime=",mintime)
maxtime = min(maxptime,maxp2time,maxwtime)
print("maxtime=",maxtime)

mintimed = int(mintime/(24*60*60))
print("mintimed=",mintimed)
maxtimed = int(maxtime/(24*60*60))
print("maxtimed=",maxtimed)

#Read the timestamp of the pressure1 data into pressure_dates 
pressure_dates = pressure_data_array[:,2].astype(np.float64)
#We are going to round all the timestamps to 2 decimal places because this makes it easier for us to correlate the timestamps 
pressure_dates = pressure_dates.round(2)  
print("pressure_dates=",pressure_dates)

#Read the timestamp of the pressure2 data into pressure2_dates 
pressure2_dates = pressure2_data_array[:,2].astype(np.float64)
pressure2_dates = pressure2_dates.round(2)  
print("pressure2_dates=",pressure2_dates)

#Read the timestamp of the wind data into wind_dates 
wind_dates = wind_data_array[:,2].astype(np.float64)
wind_dates = wind_dates.round(2) 
print("wind_dates=",wind_dates)
 
combined_dates = pd.date_range(start=mintime, end=maxtime)
print("combined_dates=",combined_dates)

#All of the file columns are read as strings 
#We need to change all the columns to the right data type which is float 64 bits 

#Read the pressure1 data into pressure_data as floats 
pressure_data =  pressure_data_array[:,5].astype(np.float64)
print("pressure_data=",pressure_data)

#Read the pressure 2 data into pressure2_data as floats 
pressure2_data =  pressure2_data_array[:,5].astype(np.float64)
print("pressure2_data=",pressure2_data)

#Read the wind data into wind_data as floats 
wind_data =  wind_data_array[:,5:9].astype(np.float64)
print("wind_data=",wind_data)

#Now we are going to create a pandas dataframe which has 2 columns 
#2 columns: data timestamp & value which is the actual pressure 
#The index of the pandas data frame is the timestamp 
pressure_df = pd.DataFrame({'datetimestamp': pressure_dates, 'value': pressure_data}).set_index('datetimestamp')
print("pressure_df=",pressure_df)

#Create pandas data frame for pressure2 data where the index is the timestamp & the pressure2 data is stored in value 
pressure2_df = pd.DataFrame({'datetimestamp': pressure2_dates, 'value': pressure2_data}).set_index('datetimestamp')
print("pressure2_df=",pressure2_df)

#Create pandas data frame for wind data where the index is the timestamp & has 4 columns: wind direction, wind speed, and U and V 
wind_df = pd.DataFrame({'datetimestamp': wind_dates, 'dir': wind_data_array[:,5].astype(np.float64), 'speed':wind_data_array[:,6].astype(np.float64) , 'u':wind_data_array[:,7].astype(np.float64), 'v':wind_data_array[:,8].astype(np.float64) }).set_index('datetimestamp')
print("wind_df=",wind_df)

#If we have pressure data for the same hundreth of a second, just retain one of them 
pressure_df = pressure_df[~pressure_df.index.duplicated()]
#We do the same for pressure2 & wind 
pressure2_df = pressure2_df[~pressure2_df.index.duplicated()]
wind_df = wind_df[~wind_df.index.duplicated()]

print("pressure_df=",pressure_df)
print("pressure2_df=",pressure2_df)
print("wind_df=",wind_df)

#We are going to corrleate & standardize all of the 3 columns using the wind dates 
#Hence, we create the standard index dates as the wind dates 
index_dates = wind_df.index.values
print("index_dates=",index_dates)


#wind_df = wind_df.reindex(index_dates).interpolate(method='polynomial', order=5)
#wind_df = wind_df.reindex(index_dates).interpolate(method='linear')
#pressure_df = pressure_df.reindex(index_dates).interpolate(method='linear')

#We are going to reindex the pressure dataframe with the index dates (wind dates) 
#Pandas reindex will correlate the two dates and calculate the pressure data for the wind dates 
#which will be slightly different from the pressure dates 
pressure_df = pressure_df.reindex(index_dates)

#Same thing for pressure2 values 
pressure2_df = pressure2_df.reindex(index_dates)
print("pressure_df=",pressure_df)
print("pressure2_df=",pressure2_df)

#Now that we have correlated pressure1, pressure2, & wind data on timestamps (standardized), 
#We can now create a combined dataframe of the 3 values
#And we use the index dates as the index  
combined_df = pd.DataFrame({
    'pressure': pressure_df['value'],
    'pressure2': pressure2_df['value'],
    'dir': wind_df['dir'],
    'speed': wind_df['speed'],
    'u': wind_df['u'],
    'v': wind_df['v']
},index=index_dates)
#}).set_index(wind_df['datetimestamp'],inplace=True)

#Drop all the columns that have 'nan' or not a num values 
combined_df = combined_df.dropna()

#Write the dataframe into pressurewindPandas.csv 
print("combined_df=",combined_df)
combined_df.to_csv('pressurewindPandas.csv')

#The pressurewindPandas.csv will have the following columns: TimeStamp, pressure1, pressure2, direction, speed, U, V.  
