#!/usr/bin/env bash

set -euo pipefail

mkdir -p examples/data

python3 <<'PY'
import csv
from pathlib import Path

header = [
    "geonameid",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature class",
    "feature code",
    "country code",
    "cc2",
    "admin1 code",
    "admin2 code",
    "admin3 code",
    "admin4 code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification date",
]

rows = [
    ["2643743", "London", "London", "", "51.5085", "-0.1257", "P", "PPLC", "GB", "", "ENG", "GLA", "", "", "8961989", "", "", "Europe/London", "2024-01-01"],
    ["5128581", "New York City", "New York City", "", "40.7143", "-74.0060", "P", "PPLA", "US", "", "NY", "061", "", "", "8804190", "", "", "America/New_York", "2024-01-01"],
    ["2988507", "Paris", "Paris", "", "48.8534", "2.3488", "P", "PPLC", "FR", "", "11", "075", "", "", "2138551", "", "", "Europe/Paris", "2024-01-01"],
    ["1850147", "Tokyo", "Tokyo", "", "35.6895", "139.6917", "P", "PPLC", "JP", "", "40", "13101", "", "", "13929286", "", "", "Asia/Tokyo", "2024-01-01"],
    ["2147714", "Sydney", "Sydney", "", "-33.8679", "151.2073", "P", "PPLA", "AU", "", "NSW", "", "", "", "5312163", "", "", "Australia/Sydney", "2024-01-01"],
    ["6058560", "London", "London", "", "42.9834", "-81.2330", "P", "PPLA", "CA", "", "ON", "", "", "", "422324", "", "", "America/Toronto", "2024-01-01"],
    ["6295630", "Earth Observation Point", "Earth Observation Point", "", "0.0000", "0.0000", "S", "OBS", "ZZ", "", "", "", "", "", "0", "", "", "UTC", "2024-01-01"],
]

path = Path("examples/data/geonames.csv")
with path.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(header)
    writer.writerows(rows)
PY

cargo run --example build-float-doctest-tree-rkyv_08 --features rkyv_08
cargo run --example build-immutable-doctest-tree-rkyv_08 --features rkyv_08
cargo run --example build-geonames-embedded-rkyv --features csv,rkyv_08
