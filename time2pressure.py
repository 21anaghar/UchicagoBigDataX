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
# load dataset
#iris = load_iris()
#irx, iry = stdsc.fit_transform(iris.data), iris.target
with open("input.csv",'r') as dest_f:
    data_iter = csv.reader(dest_f,
                           delimiter = ',',
                           quotechar = '"')
    data = [data for data in data_iter]
data_array = np.asarray(data)
print("data_array=",data_array)
irx = data_array[:, 1]
irx = irx[:1000]
irxnew = np.empty([irx.shape[0], 6])
for elem in irx:
    arr = elem.split('T')
    arr1 = arr[0].split('-')
    year = int(arr1[0])
    month = int(arr1[1])
    day = int(arr1[2])
    arr2 = arr[1].split(':')
    hour = int(arr2[0])
    minute = int(arr2[1])
    second = float(arr2[2])
    print("array entry=",[year,month,day,hour,minute,second])
    np.append(irxnew, [year,month,day,hour,minute,second], axis=None)


iry = data_array[:, 2].astype(np.float64)
iry = iry[:1000]

print("irxnew shape:", irxnew.shape)
print("iry shape:", iry.shape)
print("irxnew:", irxnew)
print("iry:", iry)
x_train, x_test, y_train, y_test = train_test_split(irxnew, iry, test_size=0.2)


# built model and train
model = elm.elm(hidden_units=32, activation_function='sigmoid', random_type='normal', x=x_train, y=y_train, C=1, elm_type='reg')
beta, train_score, running_time = model.fit('no_re')
print("regression beta:\n", beta)
print("regression train score:", train_score)
print("regression running time:", running_time)

# test
prediction = model.predict(x_test)
print("regression result:", prediction.reshape(-1, ))
print("regression score:", model.score(x_test, y_test))
prediction = prediction.reshape(-1, )

score = r2_score(y_test, prediction)
print("The accuracy of our model is {}%".format(round(score, 2) *100))

score = mean_absolute_error(y_test, prediction)
print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))


score = np.sqrt(mean_absolute_error(y_test, prediction))
print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))

# plot
#plt.plot(x_test, y_test)
#plt.plot(x_test, prediction)
#plt.title('x_test, y_test')
#plt.legend()
#plt.show()

######## Anomalies section #############################
minpressure = np.min(prediction)
print("minpressure =",minpressure)
if minpressure < 1000:
   print("Anomaly: pressure situation is possibly malignant for humans")
if minpressure < 980:
   print("Anomaly: pressure situation in dire condition. Possible severe weather storm")
