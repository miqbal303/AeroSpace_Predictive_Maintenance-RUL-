# Remaining Useful Life (RUL) Prediction for Turbofan Engines

### Objective

The objective of this project is to develop a machine learning model to predict the remaining useful life of aircraft turbofan engines. The Remaining Useful Life (RUL) is the amount of cycles an engine has left before it needs maintenance.

![sensor](https://github.com/Romachauhan1607/Pred_maintainance-Project/assets/90463649/81d74b8f-374e-48ee-9a14-316d71a256a2)

### AWS Deployment Link:

AWS deployment link: http://predictive-env-1.eba-5bwnxpqm.ap-south-1.elasticbeanstalk.com/

![Screenshot 2023-09-27 003828](https://github.com/Romachauhan1607/Pred_maintainance-Project/assets/90463649/9515587a-0f62-4d03-959e-4ea31e48f2b4)

### Project Description :

In industry, prognostics and health management are key topics for anticipating asset state and avoiding downtime and breakdowns. Run-to-Failure simulation data from turbofan jet engines is included.

The C-MAPSS software was used to simulate engine degradation. Four separate sets of operational conditions and fault modes were simulated in four different ways. To characterize fault progression, record numerous sensor channels. The Prognostics CoE at NASA Ames provided the data set.

#### Abstract

❏ NASA released dataset of 218 turbofan
engines in 2008. The data was recorded until
the point of breakdown.

❏ In this project, we would try to understand
the data-set, and predict Remaining Useful
Life (RUL) of engines from testing data.

❏ We would try to deploy several machine
learning models on this data-set, and discuss
their performance. And finally, we would try
to solve this problem using a mix-match of
various algorithms to the best of our knowledge

### Data Set Organization

Data Set: FD001 Train trjectories: 100 Test trajectories: 100 Conditions: ONE (Sea Level) Fault Modes: ONE (HPC Degradation)

Data Set: FD002 Train trjectories: 260 Test trajectories: 259 Conditions: SIX Fault Modes: ONE (HPC Degradation)

Data Set: FD003 Train trjectories: 100 Test trajectories: 100 Conditions: ONE (Sea Level) Fault Modes: TWO (HPC Degradation, Fan Degradation)

Data Set: FD004 Train trjectories: 248 Test trajectories: 249 Conditions: SIX Fault Modes: TWO (HPC Degradation, Fan Degradation)

There are 4 dataset . In this repo I work on only First dataset


### EDA

❏ Train1 shape: (20631, 26)

❏ Test1 shape: (13096, 26)

❏ Rul1 shape: (100, 1)

❏ Train1 have data from 100 engine.

❏ Training data had following features:

    ❏ Unit number of engine

    ❏ nth cycle in operation

    ❏ 3 operational settings

    ❏ 21 sensor’s readings

    ❏ Each unit number was associated with a specific engine


operational settings does not empact to predict Rul of engine and Few sensor have constant data.


### Visualization:

![1696065810081](image/Readme/1696065810081.png)


Corelation Heatmap

![1696066023415](image/Readme/1696066023415.png)

After Dropping Constant Column

![1696067057045](image/Readme/1696067057045.png)


### RESUTS:

![1696067553280](image/Readme/1696067553280.png)

![1696068219543](image/Readme/1696068219543.png)


![home](https://github.com/Romachauhan1607/Pred_maintainance-Project/assets/90463649/14fe98e3-ed05-4e26-bac9-36b0343eca33)

![Screenshot 2023-09-27 004101](https://github.com/Romachauhan1607/Pred_maintainance-Project/assets/90463649/44b4f740-5d46-4b8d-933d-b4e763025fe3)

![Screenshot 2023-09-27 004047](https://github.com/Romachauhan1607/Pred_maintainance-Project/assets/90463649/2e03be4a-0930-40e7-89db-1e76e3d20f1f)

![Screenshot 2023-09-27 004335](https://github.com/Romachauhan1607/Pred_maintainance-Project/assets/90463649/9745625a-1546-4d8c-b3bf-12c39e4c3146)

![warning](https://github.com/Romachauhan1607/Pred_maintainance-Project/assets/90463649/e6e557c6-8beb-4bb6-941c-52c8bdf259c1)

![danger'](https://github.com/Romachauhan1607/Pred_maintainance-Project/assets/90463649/5c36ead3-b910-4faa-9d3f-55b82220125f)

![1696068252168](image/Readme/1696068252168.png)


### INSTALLATION STEPS:

```
conda create -p venv python==3.9 -y
```

```
conda activate venv/
```

```
pip install -r requirements.txt
```

### Deployment:

```
python application.py
```

This is a Group Project. If anybody have any query , contact on below mails

ayushgandhi904@gmail.com

miqbal303@gmail.com

roma.chauhan@gmail.com

varshamahato1999@gmail.com


### Thanks
