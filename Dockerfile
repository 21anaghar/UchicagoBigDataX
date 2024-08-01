FROM python:3
WORKDIR /usr/src/app
COPY * .
RUN pip install numpy
RUN pip install scipy
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install matplotlib
RUN pip install influxdb_client
CMD [ "python", "./speedvspressureRT.py" ]
