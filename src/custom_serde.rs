#[cfg(feature = "serialize")]
pub(crate) mod array {
    use core::option::Option::None;
    use serde::{
        de::{SeqAccess, Visitor},
        ser::SerializeTuple,
        Deserialize, Deserializer, Serialize, Serializer,
    };
    use std::marker::PhantomData;

    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(
        data: &[T; N],
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let mut s = ser.serialize_tuple(N)?;
        for item in data {
            s.serialize_element(item)?;
        }
        s.end()
    }

    struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

    impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
    where
        T: Copy + Default + Deserialize<'de>,
    {
        type Value = [T; N];

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str(&format!("an array of length {}", N))
        }

        #[inline]
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            // can be optimized using MaybeUninit
            //let mut data = Vec::with_capacity(N);
            let mut data = [T::default(); N];

            for idx in 0..N {
                match (seq.next_element())? {
                    Some(val) => {
                        // data.push(val);
                        data[idx] = val;
                    }
                    None => return Err(serde::de::Error::invalid_length(N, &self)),
                }
            }

            Ok(data)
        }
    }
    pub fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<[T; N], D::Error>
    where
        D: Deserializer<'de>,
        T: Copy + Default + Deserialize<'de>,
    {
        deserializer.deserialize_tuple(N, ArrayVisitor::<T, N>(PhantomData))
    }
}

/*#[cfg(feature = "serialize")]
pub(crate) mod array_of_2ples {
    use core::option::Option::None;
    use serde::{
        de::{SeqAccess, Visitor},
        ser::SerializeTuple,
        Deserialize, Deserializer, Serialize, Serializer,
    };
    use std::marker::PhantomData;

    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(
        data: &[(T, T); N],
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let mut s = ser.serialize_tuple(N * 2)?;
        for item in data {
            s.serialize_element(&item.0)?;
            s.serialize_element(&item.1)?;
        }
        s.end()
    }

    struct Array2PleVisitor<T, const N: usize>(PhantomData<T>);

    impl<'de, T, const N: usize> Visitor<'de> for Array2PleVisitor<T, N>
    where
        T: Copy + Default + Deserialize<'de>,
    {
        type Value = [(T, T); N];

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str(&format!("an array of 2ples of length {}", N))
        }

        #[inline]
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            // can be optimized using MaybeUninit
            // let mut data: Vec<(T, T)> = Vec::with_capacity(N);
            let mut data = [(T::default(), T::default()); N];

            for idx in 0..N {
                match (seq.next_element::<T>())? {
                    Some(val_0) => match (seq.next_element::<T>())? {
                        Some(val_1) => {
                            // data.push((val_0, val_1));
                            data[idx].0 = val_0;
                            data[idx].1 = val_1;
                        }
                        None => return Err(serde::de::Error::invalid_length(N, &self)),
                    },
                    None => return Err(serde::de::Error::invalid_length(N, &self)),
                }
            }
            // match data.try_into() {
            //     Ok(arr) => Ok(arr),
            //     Err(_) => unreachable!(),
            // }

            Ok(data)
        }
    }
    pub fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<[(T, T); N], D::Error>
    where
        D: Deserializer<'de>,
        T: Copy + Default + Deserialize<'de>,
    {
        deserializer.deserialize_tuple(N * 2, Array2PleVisitor::<T, N>(PhantomData))
    }
}*/

#[cfg(feature = "serialize")]
pub(crate) mod array_of_arrays {
    use core::option::Option::None;
    use serde::{
        de::{SeqAccess, Visitor},
        ser::SerializeTuple,
        Deserialize, Deserializer, Serialize, Serializer,
    };
    use std::marker::PhantomData;

    pub fn serialize<S: Serializer, T: Serialize, const N: usize, const K: usize>(
        data: &[[T; K]; N],
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let mut s = ser.serialize_tuple(N * K)?;
        for item in data {
            for x in 0..K {
                s.serialize_element(&item[x])?;
            }
        }
        s.end()
    }

    struct ArrayArrayVisitor<T, const N: usize, const K: usize>(PhantomData<T>);

    impl<'de, T, const N: usize, const K: usize> Visitor<'de> for ArrayArrayVisitor<T, N, K>
    where
        T: Copy + Default + Deserialize<'de>,
    {
        type Value = [[T; K]; N];

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str(&format!("an array of arrays, dimensions of {}x{}", K, N))
        }

        #[inline]
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            // can be optimized using MaybeUninit
            // let mut data: Vec<(T, T)> = Vec::with_capacity(N);
            let mut data = [[T::default(); K]; N];

            for idx in 0..N {
                for x in 0..K {
                    match (seq.next_element::<T>())? {
                        Some(val) => {
                            data[idx][x] = val;
                        },
                        None => return Err(serde::de::Error::invalid_length(N, &self)),
                    }
                }
            }

            Ok(data)
        }
    }
    pub fn deserialize<'de, D, T, const N: usize, const K: usize>(deserializer: D) -> Result<[[T; K]; N], D::Error>
    where
        D: Deserializer<'de>,
        T: Copy + Default + Deserialize<'de>,
    {
        deserializer.deserialize_tuple(N * K, ArrayArrayVisitor::<T, N, K>(PhantomData))
    }
}
