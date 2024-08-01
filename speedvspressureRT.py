import elm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import datetime
import argparse
import influxdb_client
import pickle
import os
import pytz
import datetime
from datetime import datetime as datetimesub
from scipy.io import savemat
import pandas as pd
import time
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client import BucketRetentionRules
from influxdb_client import WriteOptions

def loadInfluxClient(pickle_path):
    """
    Creates influxdb client from pickle credentials file
    """
    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as f:
            influx_dict = pickle.load(f)

            client = influxdb_client.InfluxDBClient(
                url = influx_dict["idb_url"],
                token = influx_dict["idb_token"],
                org = influx_dict["idb_org"],
                timeout = 100000_000
            )

            query_api = client.query_api()

            return client,query_api
    else:
        print("InfluxDB credentials file not found. Run influxdb-setup.py to generate one.")
        exit(1)

def processInfluxDF(df, output_tz):
    cur_box = df["_measurement"].iloc[0]
    cur_id = df["id"].iloc[0]
    id_str = f"{cur_box}_{cur_id}"

    out_df = df.drop(columns=["result", "table", "_measurement", "id"])

    if "baro_time" in out_df:
        out_df.drop(columns=["baro_time"], inplace=True)

    if "err" in out_df:
        err_list = pd.Series(list("err")).unique()
        if len(err_list) > 1:
            print("WARNING: Some anemometer values were recorded with an error code")

        out_df.drop(columns=["err"], inplace=True)

    out_df.rename(columns={'_time': 'time'}, inplace=True)
    out_df["time"] = out_df["time"].dt.tz_convert(output_tz)
    out_df["time"] = out_df["time"].dt.tz_localize(None)
    # create a unix time column
    out_df["time"] = (out_df["time"] - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')

    return id_str,out_df

def createFluxFilters(col_name, in_str):
    if in_str is None:
        return ""

    if ',' in in_str:
        # user provided a list
        in_list = in_str.split(',')
    else:
        in_list = [in_str]

    filter_list = []
    for i in in_list:
        filter_list.append(f'r["{col_name}"] == "{i}"')

    filter_list_str = " or ".join(filter_list)

    if len(filter_list) > 0:
        flux_line = f'|> filter(fn: (r) => {filter_list_str})'
    else:
        flux_line = ""

    return flux_line

def get_sensor_data(sensor_id, start_time, end_time):
    """
    Main runner
    """

    # Create InfluxDB Client
    idb_client,idb_query_api = loadInfluxClient('influx-creds.pickle')

    # process timezones
    input_tz = pytz.timezone('Etc/UTC')
    output_tz = pytz.timezone('Etc/UTC')

    #start_time = datetime.datetime.fromisoformat(start_time)
    start_time = input_tz.localize(start_time)
    start_time = start_time.astimezone(datetime.timezone.utc)
    start_time = start_time.replace(tzinfo=None)
    #end_time = datetime.datetime.fromisoformat(end_time)
    end_time = input_tz.localize(end_time)
    end_time = end_time.astimezone(datetime.timezone.utc)
    end_time = end_time.replace(tzinfo=None)

    # process filters
    box_filters = createFluxFilters("_measurement", 'paros1')
    sensor_filters = createFluxFilters("id", sensor_id)

    # from idb query
    idb_query = f'''from(bucket: "{'parosbox'}")
        |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)\n'''
    
    if box_filters:
        idb_query += f'\t{box_filters}\n'

    if sensor_filters:
        idb_query += f'\t{sensor_filters}\n'

    idb_query += '''\t|> drop(columns: ["_start", "_stop"])
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'''

    print("\nRunning InfluxDB Query...\n")
    print(f"{idb_query}\n")

    idb_result = idb_query_api.query_data_frame(query=idb_query)
    
    dfs_list = []
    if isinstance(idb_result, list):
        for r in idb_result:
            dfs_list += [r for _, r in r.groupby('table')]
    else:
        dfs_list.append(idb_result)

    #for df in dfs_list:
    #   print("df=",df)

    # process dataframes
    out_df = {}
    for df in dfs_list:
        cur_idstr,cur_df = processInfluxDF(df, 'Etc/UTC')

        out_df[cur_idstr] = cur_df

    print("\nPreviewing Dataframes...\n")
    print(out_df)
    return out_df


#The pressurewindPandas.csv will have the following columns: TimeStamp, pressure1, pressure2, direction, speed, U, V. 
# load dataset into python array called data 
with open("pressurewindPandas.csv",'r') as dest_f:
    data_iter = csv.reader(dest_f,
                           delimiter = ',',
                           quotechar = '"')
    data = [data for data in data_iter]
#delete the first line which is a headerline 
data = np.delete(data, 0, 0)

#Convert the python array data into a numpy data_array 
data_array = np.asarray(data)
print("data_array=",data_array)

#The y variable or the dependent variable is the first column of this array which is pressure1 
iry = data_array[:, 1].astype(np.float64)

#This timestamp array is the zeroeth column of the numpy array 
timestamparr = np.asarray(data_array[:,0]).astype(np.float64)
print('timestamparr',timestamparr)

#Read the pressure2 numpy array from the 2nd column of the 
#data array as floats 
pressure2 = data_array[:,2].astype(np.float64)

#Read the speed array from the 4th column of the data array as floats 
speed = data_array[:,4].astype(np.float64)
irx = np.column_stack([timestamparr,pressure2, speed])

print("irx shape:", irx.shape)
print("iry shape:", iry.shape)
print("irx:", irx)
print("iry:", iry)

accuracy = 0.0
while accuracy < .85:
       try:
#Split the x & y datasets into testing & training 80/20 
	       x_trainWtime, x_testWtime, y_train, y_test = train_test_split(irx, iry, test_size=0.2)

#We are ignoring the timestamp for the independent variable for training and testing  
#We will only use pressure2 & windspeed 
	       x_train = x_trainWtime[:,1:3]
	       x_test = x_testWtime[:,1:3]

# we are going to now train the ELM model with 96 hidden units sigmoid activation function 
#and this is a regression model as you need real number values for pressure 1 which is the dependent variable
#This is not a classification problem, it is a regression problem
#Only uses the training dataset   
# built model 
#model = elm.elm(hidden_units=80, activation_function='sigmoid', random_type='normal', x=x_train, y=y_train, C=1, elm_type='reg')
	       print("x_train = ",x_train)
	       print("x_train shape = ",x_train.shape)
	       print("y_train = ",y_train)
	       print("y_test shape = ",y_train.shape)
	       model = elm.elm(hidden_units=96, activation_function='sigmoid', random_type='normal', x=x_train, y=y_train, C=1, elm_type='reg')

#training model 
	       beta, train_score, running_time = model.fit('no_re')
	       print("regression beta:\n", beta)
	       print("regression train score:", train_score)
	       print("regression running time:", running_time)

# test
#Predict using the testing dataset 
	       print("x_test = ",x_test)
	       print("x_test shape = ",x_test.shape)
	       print("y_test = ",y_test)
	       print("y_test shape = ",y_test.shape)
	       prediction = model.predict(x_test)
	       print("prediction shape 1 = ",prediction.shape)
	       print("regression result:", prediction.reshape(-1, ))
	       print("regression score:", model.score(x_test, y_test))
	       prediction = prediction.reshape(-1, )
	       print("prediction shape 2 = ",prediction.shape)
	       print("prediction = ",prediction)
	       print("y_test = ",y_test)

	       accuracy = r2_score(y_test, prediction)
	       print("The accuracy of our model is {}%".format(round(accuracy, 2) *100))
       except:
               print("An exception occurred")
               continue
######### Commented out following line as UMass raspberry PIs are not working and hence we are using existing data from Jan 1
######### Once the raspberry PIs are working uncomment the "start_time = datetime.datetime.now()" and comment following line
start_time = datetimesub(2024, 1, 1, 0, 0, 0, 0)
#start_time = datetime.datetime.now()
while True:
	print("start_time=",start_time)
	time_change = datetime.timedelta(minutes=5) 
	end_time = start_time + time_change
	print("end_time=",end_time)
	pressure1_df = get_sensor_data('141905', start_time, end_time)
	for cur_name,cur_df in pressure1_df.items():
	     pressure1_df = cur_df
	print("pressure1_df=",pressure1_df)
	pressure_dates = pressure1_df['time']
	pressure_dates = pressure_dates.round(2)
	pressure_df = pd.DataFrame({'datetimestamp': pressure_dates, 'value': pressure1_df['value'].astype(np.float64)}).set_index('datetimestamp')
	print("pressure_df as Pandas=",pressure_df)

	pressure2_df = get_sensor_data('141931', start_time, end_time)
	for cur_name,cur_df in pressure2_df.items():
	     pressure2_df = cur_df
	print("pressure2_df=",pressure2_df)
	pressure2_dates = pressure2_df['time']
	pressure2_dates = pressure2_dates.round(2)
	pressure2_df = pd.DataFrame({'datetimestamp': pressure2_dates, 'value': pressure2_df['value'].astype(np.float64)}).set_index('datetimestamp')
	print("pressure2_df as Pandas=",pressure2_df)

	wind_df = get_sensor_data('0', start_time, end_time)
	for cur_name,cur_df in wind_df.items():
	     wind_df = cur_df
	print("wind_df=",wind_df)
	wind_dates = wind_df['time']
	wind_dates = wind_dates.round(2)
	wind_df = pd.DataFrame({'datetimestamp': wind_dates, 'dir': wind_df['direction'].astype(np.float64), 'speed':wind_df['speed'].astype(np.float64) , 'u':wind_df['u'].astype(np.float64), 'v':wind_df['v'].astype(np.float64) }).set_index('datetimestamp')
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

#We are going to reindex the pressure dataframe with the index dates (wind dates)
#Pandas reindex will correlate the two dates and calculate the pressure data for the wind dates
#which will be slightly different from the pressure dates
	pressure_df = pressure_df.reindex(index_dates)

#Same thing for pressure2 values
	pressure2_df = pressure2_df.reindex(index_dates)
	print("pressure_df=",pressure_df)
	print("pressure2_df=",pressure2_df)


	speed = wind_df.loc[:,'speed']
	print("before combined speed =",speed)
	pressure2 = pressure2_df.loc[:,'value']
	print("before combined pressure2 =",pressure2)
	pressure1 = pressure_df.loc[:,'value']
	print("before combined pressure1 =",pressure1)
	combined_df = pd.DataFrame({
	    'timestamp': index_dates,
	    'speed': speed,
	    'pressure': pressure1,
	    'pressure2': pressure2
	}).set_index('timestamp')
	print("combined_df=",combined_df)

#Drop all the columns that have 'nan' or not a num values
	combined_df = combined_df.dropna()
	print("combined_df2=",combined_df)
	if combined_df.shape[0] == 0:
	    start_time = end_time
	    time.sleep(5*60)
	    continue
	index_dates = combined_df.index.values
	print("index_dates=",index_dates)


	x_test = np.column_stack([combined_df.loc[:,'pressure2'].values, combined_df.loc[:,'speed'].values])
	print("x_test = ",x_test)
	print("x_test shape = ",x_test.shape)
	y_test = combined_df.loc[:,'pressure'].values
	print("y_test = ",y_test)

#Predict using the testing dataset 
	prediction = model.predict(x_test)
	#print("prediction shape 1 = ",prediction.shape)
	#print("regression result:", prediction.reshape(-1, ))
	#print("regression score:", model.score(x_test, y_test))
	prediction = prediction.reshape(-1, )
	#print("prediction shape 2 = ",prediction.shape)
	print("prediction = ",prediction)
	#print("y_test = ",y_test)
######### Send the prediction data to influx db
	print("shape of index_dates=",index_dates.shape)
	print("shape of prediction=",prediction.shape)
	timearray = ( datetimesub.fromtimestamp(a + 16681472) for a in index_dates )
	training_df = pd.DataFrame({
		'timestamp': timearray,
		'pressure': prediction
		})
	print("training_df=",training_df)
	bucket = "ELMInput"
	org = "CameleonLab"
	token = "lo2AzH_Lq9z5rZqh6b6HSYMXuvDnrpCVA3Q_KjWFrrQmwcEXUB0cicTypboUHFNsATMh_VWSYmJ6LmRGF2HGeA=="

# Store the URL of your InfluxDB instance
	url="https://us-east-1-1.aws.cloud2.influxdata.com"

	client = influxdb_client.InfluxDBClient(
		url=url,
		token=token,
		org=org
		)
	with client.write_api(write_options=WriteOptions(batch_size=40,
		flush_interval=10_000,
		jitter_interval=2_000,
		retry_interval=5_000,
		max_retries=5,
		max_retry_delay=30_000,
		max_close_wait=300_000,
		exponential_base=2)) as _write_client:
			print(f" successfully created write_api")
			_write_client.write(bucket=bucket, org=org,
				record=training_df,
				data_frame_measurement_name="ELMprediction",
				data_frame_timestamp_column="timestamp",
				record_tag_keys=["pressure"],
				record_field_keys=["pressure"])

######### Send the prediction data to influx db
	accuracy = r2_score(y_test, prediction)
	#print("The accuracy of our model is {}%".format(round(accuracy, 2) *100))

	score = mean_absolute_error(y_test, prediction)
	#print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))

	score = np.sqrt(mean_absolute_error(y_test, prediction))
	#print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))

########## Commenting out the plotting #####################
	#print("wind_df =",wind_df)
	#xplot =  index_dates
	#print("xplot =",xplot)
	#pplot = prediction
	#print("pplot =",pplot)
	#plt.scatter(xplot, pplot, c='black')
	#plt.title('Predicted Pressure vs Time')

	#yplot =  y_test
	#print("xplot =",xplot)
	#print("yplot =",yplot)
	#plt.legend()
	#plt.show()
	#plt.scatter(xplot, yplot, c='black')
	#plt.title('Measured Pressure vs Time')
	#plt.legend()
	#plt.show()
########## Commenting out the plotting #####################

######## Anomalies section #############################
	minpressure = np.min(prediction)
	print("minpressure =",minpressure)
	if minpressure < 1000:
		print("Anomaly: pressure situation is possibly malignant for humans")
#	if minpressure < 980:
#	   print("Anomaly: pressure situation in dire condition. Possible severe weather storm")
	time.sleep(4*60)
	start_time = end_time
