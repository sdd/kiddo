# index-less stem nodes

           01                       1    2N, 2N+1
     02         03                  2    2N, 2N+1
  04   05     06   07               4    2N, 2N+1
08 09 10 11 12 13 14 15             8

    (0)
    01
 02    03
0  1  2  3

* this implies 16 - 31 are leaf nodes. If, however, a stem node split plane val is NaN, this implies that we should look at a leaf node instead.
* The LSB of the final row of nodes could be used as a niche to indicate whether the children are dynamic stem nodes, or leaf nodes, obv at the loss of 1 bit of precision (which may not be noticeable as the bucket likely covers more than one bit of dynamic range and will have the full position range).
* when setting the split value of the last row of nodes, initially always choose a split value with the LSB clear. this indicates that the children
* are actual leaves.
* if a leaf fills and needs splitting, and its parent is a main stem, set the LSB of the main stem's split plane, to indicate that the children will be dynamic stem nodes.
* when dynamic stem nodes are first required, allocate a Vec<> of size 16 in the case above. static stem child indices are assumed to be (2N - 16) and (2N - 15).
* the dynamic stem nodes can be assumed to use the MSB to indicate whether their child is a leaf node or another dynamic node. a SET MSB indicates a leaf node child, of index n-MSB, as per kiddo v2.
* when a top-level dstem is required, two must be created
* leaf nodes are initially a vec with capacity of log2(cap).
* use maybeuninit and Vec::from_raw_parts when creating the leaf vec



If 08-15 are attempted to be split?

1) just allocate 16-31 and continue
2) rebalance the tree
3) introduce a dynamic stem layer that does contain child node indices

initial state:
    Stem split vals: [NaN, ...]
    dynamic stem nodes: None
    leaf nodes: [0-15 exist but only 0 is initialized]

when leaf 0 splits:
    Stem split vals: [n, NaN, ...]
    dynamic stem nodes: None
    leaf nodes: [0-15 exist but only 0 and 8 initialized]

if leaf 0 now splits again:
    Stem split vals: [n, n, NaN, ...]
    dynamic stem nodes: None
    leaf nodes: [0-15 exist but only 0, 4, and 8 initialized]

if leaf 0 now splits again:
    Stem split vals: [n, n, NaN, n, NaN, ...]
    dynamic stem nodes: None
    leaf nodes: [0-15 exist but only 0, 2, 4, and 8 initialized]

if leaf 0 now splits again:
    Stem split vals: [n, n, NaN, n, NaN, NaN, NaN, n, Nan...]
    dynamic stem nodes: None
    leaf nodes: [0-15 exist but only 0, 1, 2, 4, and 8 initialized]


### 32/455191

#### Desired Qty

                   01:32
         02:31             03:1
   04:16    05:15      06:1     07:0
08:8 09:8 10:7 11:8 12:1 13:0 14:0 15:0


#### Desired Shift

                   01:1
         02:0             03:0
   04:0      05:1      06:0     07:0
08:0 09:0 10:1 11:0 12:0 13:0 14:0 15:0




4 4 4 4
