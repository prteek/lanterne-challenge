# lanterne-challenge
Lanterne predictive modelling challenge

## Challenge Description

The challenge is to **predict demand for micromobility vehicles over time and space**. This is a **regression problem** where Y is bounded between 0 and infinity.

You have **two hours** to work on the challenge. Don't be concerned if you do not complete everything you would like to.

## Dataset

You are given data on demand in a city across nodes and timesteps, with a 4 hour interval. Each node is an area in the city.

[train.csv](./train.csv)

[test.csv](./test.csv)

### Columns

- **timestamp**: the timestamp for the beginning of the 4 hour time interval
- **node_id**: The id of the node / area in the city
- **demand**: the value for demand that has been observed over the next 4 hours from the timestamp
- **travelling_proportion**: proportion of people travelling during the corresponding time of day
- **tempC**: temperature in degrees Celcius
- **precipMM**: precipitation in mm
- **station_count**: number of bus stops, train stations etc. around the area in which demand was observed
- **count_retail/commercial**: number of retail/commercial points of interest around the area in which demand was observed
- **cover_retail/commercial**: value indicating the presence of these as a proportion of the total area in which demand was observed
- **distance_to_**: the distance to these areas from the area in which demand was observed (in metres)
- **cycleway/tertiary/secondary_connectivity**: the number of each of these road types passing through the area in which demand was observed
- **tourist_attractions**: number of tourist attractions in the area in which demand was observed.


## To train the model and save (edit/or as it is) use ```train.py```
```shell
python train.py

```
