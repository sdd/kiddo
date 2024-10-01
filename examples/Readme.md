# Examples

## Example 1: Cities ([cities.rs](./cities.rs))

### Prerequisites

This example uses the World Cities Database dataset, created by SimpleMaps.com and downloadable from Kaggle. You'll need to download it into this `examples` folder and unzip it in order to use these examples. You can find it here: [https://www.kaggle.com/datasets/juanmah/world-cities?resource=download&select=worldcities.csv](https://www.kaggle.com/datasets/juanmah/world-cities?resource=download&select=worldcities.csv).

These examples demonstrate the creation of a kd-tree, populated by data on world cities. It demonstrates the following:

* Conversion of lat/long co-ordinates into 3D unit sphere co-ordinates
* Construction and population of a Kiddo kd-tree containing the positions of the cities in the dataset
* Serde-based deserialization from CSV
* Querying the nearest single city to a specific point on Earth using `nearest_one`
* Finding the nearest five cities to a specific pint on Earth, ordered by distance, using `nearest_n`
* Using `within` to find all cities within a specified radius of a certain point on Earth
* Finding the three most populous cities within 1000km of a certain point o Earth by using `best_n_within`




## Example 2: High-speed zero-copy serialization and deserialization with Rkyv ([rkyv.rs](./rkyv.rs))

* Rkyv-based blazingly fast serialization / deserialization

The output below was from the same machine as the serde example above -
you can see the tremendous speed improvement that can be had by switching to
Rkyv for serialization / deserialization

```
> cargo run --release --example rkyv --features="rkyv"
   Compiling kiddo v3.0.0 (~/kiddo)
    Finished release [optimized + debuginfo] target(s) in 9.71s
     Running `target/release/examples/rkyv`
Cities successfully parsed from CSV: 11061987
Parsed 11061987 rows from the CSV: (3.97 s)
Populated kd-tree with 11061987 items (1.19 s)

Nearest city to 52.5N, 1.9W: CityCsvRecord { name: "Aston", lat: 52.5, lng: -1.88333 }
Serialized kd-tree to rkyv file (306.00 ms)


Memmap ZC Deserialized rkyv file back into a kd-tree (23.31 μs)
Nearest city to 52.5N, 1.9W: CityCsvRecord { name: "Aston", lat: 52.5, lng: -1.88333 }
total elapsed: 50.53 μs

```

## Example 3: Serde Serialization to binary formats ([serde.rs](./serde.rs))

* Serde-based serialization to gzipped bincode
* Serde-based deserialization from gzipped bincode
* 
### Prerequisites

This example uses the larger GeoNames database, created by geonames and downloadable from Kaggle. You'll need to download it into this `examples` folder and unzip it in order to use these examples. You can find it here: [https://www.kaggle.com/datasets/geonames/geonames-database](https://www.kaggle.com/datasets/geonames/geonames-database).

The output below was run on a Ryzen 5900X with 32Gb DDR4-3600.

```
> cargo run --release --example serde --features="serde"
   Compiling kiddo v2.0.2 (~/kiddo)
    Finished release [optimized + debuginfo] target(s) in 5.18s
     Running `target/release/examples/serde`
Cities successfully parsed from CSV: 11061987
Parsed 11061987 rows from the CSV: (3.35 s)
Populated kd-tree with 11061987 items (2.74 s)

Nearest city to 52.5N, 1.9W: CityCsvRecord { name: "Aston", lat: 52.5, lng: -1.88333 }
Serialized kd-tree to gzipped bincode file (5.72 s)
Deserialized gzipped bincode file back into a kd-tree (2.71 s)

Nearest city to 52.5N, 1.9W: CityCsvRecord { name: "Aston", lat: 52.5, lng: -1.88333 }
```
