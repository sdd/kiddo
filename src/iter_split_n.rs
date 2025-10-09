pub struct SplitN<I> {
    iter: I,
    n: usize,
    exhausted: bool,
}

pub struct FirstHalf<'a, I> {
    split: &'a mut SplitN<I>,
}

pub struct RestHalf<'a, I> {
    split: &'a mut SplitN<I>,
}

pub trait SplitNExt: Iterator + Sized {
    fn split_n(self, n: usize) -> SplitN<Self>;
}

impl<I> SplitNExt for I
where
    I: Iterator,
{
    fn split_n(self, n: usize) -> SplitN<Self> {
        SplitN {
            iter: self,
            n,
            exhausted: false,
        }
    }
}

impl<I> SplitN<I> {
    pub fn first(&mut self) -> FirstHalf<'_, I> {
        FirstHalf { split: self }
    }

    pub fn rest(&mut self) -> RestHalf<'_, I> {
        RestHalf { split: self }
    }
}

impl<'a, I> Iterator for FirstHalf<'a, I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.split.n == 0 {
            None
        } else {
            self.split.n -= 1;
            self.split.iter.next()
        }
    }
}

impl<'a, I> Iterator for RestHalf<'a, I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.split.exhausted {
            None
        } else {
            while self.split.n > 0 {
                self.split.iter.next()?;
                self.split.n -= 1;
            }
            let next = self.split.iter.next();
            if next.is_none() {
                self.split.exhausted = true;
            }
            next
        }
    }
}

impl<'a, I> RestHalf<'a, I>
where
    I: Iterator,
{
    pub fn is_empty(&mut self) -> bool {
        if self.split.exhausted {
            true
        } else {
            while self.split.n > 0 {
                if self.split.iter.next().is_none() {
                    self.split.exhausted = true;
                    return true;
                }
                self.split.n -= 1;
            }
            self.split.iter.size_hint().0 == 0
        }
    }
}
