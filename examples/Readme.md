# Examples

## Example 1: Cities

These examples demonstrate the creation of a kd-tree, populated by data on world cities. It demonstrates the following:

* Conversion of lat/long co-ordinates into 3D unit sphere co-ordinates
* Construction and population of a Kiddo kd-tree containing the positions of the cities in the dataset
* Serde-based deserialization from CSV
* Querying the nearest single city to a specific point on Earth using `nearest_one`
* Finding the nearest five cities to a specific pint on Earth, ordered by distance, using `nearest_n`
* Using `within` to find all cities within a specified radius of a certain point on Earth
* Finding the three most populous cities within 1000km of a certain point o Earth by using `best_n_within`

### Prerequisites

This example uses the World Cities Database dataset, created by SimpleMaps.com and downloadable from Kaggle. You'll need to download it into this `examples` folder and unzip it in order to use these examples. You can find it here: [https://www.kaggle.com/datasets/juanmah/world-cities?resource=download&select=worldcities.csv](https://www.kaggle.com/datasets/juanmah/world-cities?resource=download&select=worldcities.csv).



## Example 2: Serde Serialization to binary formats (TODO)

* Serde-based deserialization from JSON
* Serde-based serialization to gzipped bincode



## Example 3: Rkyv high-speed zero-copy serialization and deserialization (TODO)

* Rkyv-based blazingly fast serialization / deserialization
