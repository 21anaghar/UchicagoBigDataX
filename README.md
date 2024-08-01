Meteorologic Real Time Extreme Learning Machine Pressure Prediction on FLOTO


Abstract 

This study leverages Extreme Learning Machines (ELM) for real-time anomaly detection in atmospheric pressure data using FLOTO's Raspberry Pi devices. ELM, known for its fast training and efficiency, is applied to predict pressure anomalies, crucial for human health and weather forecasting. Data from InfluxDB, encompassing pressure and wind sensor readings, is preprocessed, with a model trained on 80% of 550,000 records, achieving a 92% R² score. The real-time implementation on FLOTO demonstrates effective anomaly detection, with results visualized via Grafana. This approach provides a cost-effective, scalable solution for monitoring and predicting hazardous atmospheric conditions.


Introduction 

Extreme Learning Machine (ELM) was introduced by Huang at. al. in 2006. It has gained much traction owing to its fast-learning, easy implementation, good generalization from limited data, and applications to real time analysis. ELM differs from the feedforward neural networks (FFN) in a few integral ways. Firstly, ELM removes multiple iterations in FFN and instead uses a single step in the learning process. While FFN uses gradient-based back-propagation, in ELM, the weights of the output layer are computed using the Moore-Penrose inverse of the hidden layers output matrix. 

FLOTO is a system that deploys and manages hundreds of single-board computers, such as Raspberry Pis, for extensive field data collection. It enables remote, secure operation of these devices without needing physical access.



Motivation 

Our main motivation is running an Extreme Learning Machine model on the Edge using FLOTO Raspberry Pi devices and detecting anomalies in real time. ELM was chosen as the ML model as it is most conducive for fast training & real time applications. Additionally, low atmospheric pressures (<1000 mbar) can be malignant for humans1 & low-cost anomaly detection will aid in predicting hazardous pressures. 




Research Methods 

 
Refer to preparedataPandas.py
    Implements ELM Training (1)--> Data Preparation (B) as follows
Refer to speedvspressureBatch.py that trains the ELM model and tests it with the testing dataset
    Implements ELM Training (1)-->Model Creation (C) as follows
Refer to speedvspressureRT.py that trains the ELM model and uses the model to make real time predictions with data from sensors via influxdb
    Implements ELM Training (1)-->Model Creation (C) 
               ELM Real Time Prediction (2)--> Data Ingestion (A), Data Preparation (B), & Model Prediction & Anomaly Detection on FLOTO (C) as follows

ELM Training (1) 

Data Ingestion (A) 

The InfluxDB is queried for data from two pressure sensors and a singular wind sensor. The data from the sensors are queried from the historical records in InfluxDB for a time period from January 1, 2024 to January 10, 2024. This produces three comma separated files with pressure at location 1, wind data at location 1, and pressure at location 2. All of these files are time stamped. 

Data Preparation (B) 

The problem with data produced in the data ingestion step is that the timestamps are skewed by fractions of a second. This needed to be corrected. The three files are read into PANDAS data frames with the relevant data types. Each of these data frames are indexed by their timestamps. The timestamps of the wind data are considered as the standardized index. Pressure 1 & Pressure 2 data frames are re-indexed using the wind data frame timestamps and interpolated within the timestamps. Hence, the three data frames are correlated and standardized onto the timestamps. We now create a combined data frame with the pressure 1, pressure 2, wind data, all indexed with the standardized timestamp. All the rows which have an ‘NAN’ are dropped. The resulting data frame has about 550,000 records. This file is fed into the model creation process as well as propagated into the InfluxDB for visualization purposes. 

Model Creation (C) 

The file created during the data preparation step has four columns: timestamp, pressure 1, pressure 2, wind speed. This file is read into a numpy data array. The independent variables are pressure 2 and wind speed and the dependent variable is pressure 1. The 550,000 records are split into training & testing data with a 80/20 split. We now train the ELM model with 96 hidden units and the sigmoid activation function. This is a regression problem and not a classification problem. All data is stored as floats. The training time averaged around a few minutes. The accuracy when the model was run on the testing data was measured using the R2_score methodology. 


ELM Real Time Prediction (2) 
	
Data Ingestion (A) 

The model created is run in a dockerized container both on the Chameleon testbed and in FLOTO. The real time program runs in a loop. It queries the InfluxDB for the pressure 2 and wind speed data from the corresponding sensors in a real time mode in a batch of a few seconds. The results are stored in PANDAS dataframes indexed by timestamp. 


Data Preparation (B) 

As in the ELM training phase data preparation, the small data frame which contains the pressure 2 & wind speed is again correlated into a common timestamp index using PANDAS. 


Model Prediction & Anomaly Detection on FLOTO (C)

The model created in the training process is used to predict pressure 1 using the independent variables, pressure 2, & wind speed. The predicted values are propagated to InfluxDB for visualization purposes. Certain pressure anomalies are automatically detected & flagged. Examples of two pressure anomalies: Below 1000 mbar (malignant to humans- associated with higher daily rates of Myocardial Infarction)2 and below 980 (dire condition; possible severe storm). 
The model prediction & anomaly detection was implemented using Python. A docker image for the Python program was created using an ARM architecture and deployed to DockerHub. This Docker image was used to create a service on FLOTO. An application was created using this deployed service and a job was scheduled for the ML application. This application was successfully tested on the FLOTO device. 

Visualization (3) 

The Grafana dashboard was used to connect to InfluxDB and the following data was visualized in the Grafana dashboard: timeseries of measured pressure 1, timeseries of measured pressure 2, & timeseries of measured wind data. The ELM predicted pressure 1 timeseries data was also visualized on the Grafana dashboard. 	


Results 

With the 550,000*0.8 record input of training data & 96 hidden units with a sigmoid activation function, the ELM model accuracy obtained was 92%. This was measured using R2_score methodology. 


Next Steps

Currently the data from the sensors is flowing into InfluxDB and into our ML model. InfluxDB is a real time database for time series data. The next step is to replace InfluxDB with a data pipeline from the Raspberry Pis connected to the sensors to the ML prediction program running on the edge.

