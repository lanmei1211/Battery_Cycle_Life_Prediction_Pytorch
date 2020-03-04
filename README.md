 	~~IN THE MAKING... CURRENTLY WORKING ON THE DATALOADER!~~ (DONE FOR TIMESERIES DATA)
  # IN THE MAKING... CURRENTLY WORKING ON THE MODEL!


# Battery life cycle predition

## Get the data

The data is available [here](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204). Download the three batches in `.mat`. Then create a `data` folder in the project's root directory and move the `.mat` there. The final folder structure should look like this:

```
long_cycle_life_prediction
├── data
|   ├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
|   ├── 2018-04-12_batchdata_updated_struct_errorcorrect.mat
|   └── 2017-06-30_batchdata_updated_struct_errorcorrect.mat
```

## The data

The dataset is comprised of 124 batteries; each of them with a life cycle (a complete charge and discharge) range between 150 and 2300 approximately. It is split into 3 subdatasets:
 * train (41 batteries)
 * test (43 batteries)
 * tets2 (40 batteries)

Batteries are charged and discharged until the battery reaches 80% of its original capacity. The number of cycles it takes for a battery to reach this state is called the battery use cycle. In the dataset this number lies between 148 and 2237 after filtering out the faulty batteries.

![](cycle_number_VS_discharge_capacity.png)

The data is comprised of scalar values associated to a specific cycle and the timeseries data corresponding to it.

The scalar values are comprised of:

 * Internal resistance (Ω)
 * Total charge (Ah)
 * Discharge time (Minutes)
 * Remaining cycles (Positive integer)
 
The timeseries data is comprised of: 

 * Temperature (°C)
 * Charge (Ah)
 * Voltage (V)
 * Current (A)
 


## Preprocessing

In order to preprocess the data run

```
python3 data_preprocessing.py
```

from the base directory. This will generate `preprocessed_data.pkl` file in your `./data` directory.The generated result is a nested dict with the following structure:

```
dataset
    b1c1
        cycle_life # total cycles untill 80% of thenomial capacity has been reached
        summary # each of these subentries contain 1 value associated to each cycle 
            IR 
            QD
            Remaining_cycles
            Discharge_time
        cycles
            1
                Qdlin
                Tdlin
                Vdlin
            2
                Qdlin
                Tdlin
                Vdlin
                
            ...
            n
            
                Qdlin
                Tdlin
                Vdlin    
    b1c2
        ...
            ...
```        


