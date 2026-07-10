use kiddo::{Eytzinger, KdTree, VecOfArenas};

#[derive(Debug, Clone, rkyv_08::Archive, rkyv_08::Serialize, rkyv_08::Deserialize)]
#[rkyv(crate = rkyv_08)]
pub struct EmbeddedCity {
    pub name: String,
    pub country_code: String,
    pub lat: f32,
    pub lng: f32,
    pub population: u32,
}

pub type Tree = KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, 3, 32>, 3, 32>;

#[derive(rkyv_08::Archive, rkyv_08::Serialize, rkyv_08::Deserialize)]
#[rkyv(crate = rkyv_08)]
pub struct EmbeddedGeoNames {
    pub tree: Tree,
    pub cities: Vec<EmbeddedCity>,
}
