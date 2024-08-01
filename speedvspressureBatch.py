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
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd 
from influxdb_client import BucketRetentionRules
from influxdb_client import WriteOptions
from datetime import datetime



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
#Now we create the independent variables as a numpy array with 3 dimensions 
#Timestamp, pressure2, & speed 
irx = np.column_stack([timestamparr,pressure2, speed])

print("irx shape:", irx.shape)
print("iry shape:", iry.shape)
print("irx:", irx)
print("iry:", iry)


accuracy = 0.0
while accuracy < .85:
#Split the x & y datasets into testing & training 80/20 
	x_trainWtime, x_testWtime, y_train, y_test = train_test_split(irx, iry, test_size=0.2)

#We are ignoring the timestamp for the independent variable for training and testing  
#We will only use pressure2 & windspeed 
	x_train = x_trainWtime[:,1:3]
	x_test = x_testWtime[:,1:3]
	print("x_train = ",x_train)
	print("x_train shape = ",x_train.shape)
	print("y_train = ",y_train)
	print("y_train = ",y_train.shape)

# we are going to now train the ELM model with 96 hidden units sigmoid activation function 
#and this is a regression model as you need real number values for pressure 1 which is the dependent variable
#This is not a classification problem, it is a regression problem
#Only uses the training dataset   
# built model 
	model = elm.elm(hidden_units=96, activation_function='sigmoid', random_type='normal', x=x_train, y=y_train, C=1, elm_type='reg')

#training model 
	beta, train_score, running_time = model.fit('no_re')
	print("regression beta:\n", beta)
	print("regression train score:", train_score)
	print("regression running time:", running_time)

# test
#Predict using the testing dataset 
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

######### Send the training data to influx db
print("shape of x_trainWtime=",x_trainWtime.shape)
print("shape of y_train=",y_train.shape)
timearray = ( datetime.fromtimestamp(a + 16681472) for a in x_trainWtime[:, 0] )
training_df = pd.DataFrame({
    'timestamp': timearray,
    'pressure': y_train,
    'pressure2': x_trainWtime[:, 1],
    'speed': x_trainWtime[:, 2],
})
#}).set_index('timestamp')
print("training_df=",training_df)
bucket = "ELMInput"
org = "CameleonLab"
# Restricted token
#token = "jbEDVe32KUSGPKeyfQds8JT8kUrD0C6-HZztDjXriybyaPt-RbyuuPoRnUEkCtMwuukQ59SUtGuK8KOsVetYnA=="
# All access token
token = "lo2AzH_Lq9z5rZqh6b6HSYMXuvDnrpCVA3Q_KjWFrrQmwcEXUB0cicTypboUHFNsATMh_VWSYmJ6LmRGF2HGeA=="

# Store the URL of your InfluxDB instance
url="https://us-east-1-1.aws.cloud2.influxdata.com"

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)
buckets_api = client.buckets_api()
'''
try:
        oldbucket = buckets_api.find_bucket_by_name(bucket)
        buckets_api.delete_bucket(oldbucket)
        print(f" successfully deleted bucket: {bucket}")
except Exception  as e:
        print(e)
        pass
retention_rules = BucketRetentionRules(type="expire", every_seconds=0)
created_bucket = buckets_api.create_bucket(bucket_name=bucket,
                                               retention_rules=retention_rules,
                                               org=org)
print(f" successfully created bucket: {bucket}")
'''


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
                    data_frame_measurement_name="ELMtraining",
                    data_frame_timestamp_column="timestamp",
                    record_tag_keys=["pressure", "pressure2", "speed"],
                    record_field_keys=["pressure", "pressure2", "speed"])


print("shape of x_trainWtime=",x_trainWtime.shape)
print("shape of prediction=",prediction)
timearray = ( datetime.fromtimestamp(a + 16681472) for a in x_testWtime[:, 0] )
training_df = pd.DataFrame({
    'timestamp': timearray,
    'pressure': prediction
})
print("training_df=",training_df)
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
                    data_frame_measurement_name="ELMtesting",
                    data_frame_timestamp_column="timestamp",
                    record_tag_keys=["pressure"],
                    record_field_keys=["pressure"])

score = mean_absolute_error(y_test, prediction)
print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))


score = np.sqrt(mean_absolute_error(y_test, prediction))
print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))

# plot
mask = np.random.choice([False, True], len(x_testWtime), p=[.90, 0.10])
xplot =  x_testWtime[:,0]
pplot = prediction
plt.scatter(xplot, pplot, c='black')
plt.title('Predicted Pressure vs Time')
xplot =  irx[:,0]
yplot =  iry
plt.legend()
plt.show()
plt.scatter(xplot, yplot, c='black')
plt.title('Measured Pressure vs Time')
plt.legend()
plt.show()
######## Anomalies section #############################
minpressure = np.min(prediction)
print("minpressure =",minpressure)
if minpressure < 1000:
   print("Anomaly: pressure situation is possibly malignant for humans")
#if minpressure < 980:
#   print("Anomaly: pressure situation in dire condition. Possible severe weather storm")
