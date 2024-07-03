use std::{marker::PhantomData, ops::Deref};

struct IterWrapper<I>(I);

pub trait Refs : Sized {
    fn refs(self) -> IterWrapper<Self>;
}

impl<I:Iterator> Refs for I {
    fn refs(self) -> IterWrapper<Self> {
        IterWrapper(self)
    }
}

struct IterWrapper2<I,D>(I, PhantomData<D>);

pub trait AlreadyRefs : Sized {
    fn already_refs<D>(self) -> IterWrapper2<Self,D>;
}

impl<I:Iterator> AlreadyRefs for I {
    fn already_refs<D>(self) -> IterWrapper2<Self,D> {
        IterWrapper2(self, PhantomData)
    }
}

pub trait RefIterator : Sized {
    type Item:?Sized;

    fn next_and<T,F>(&mut self, f: F) -> Option<T>
    where
        F: FnOnce(&Self::Item) -> T;

    fn collectz<C>(mut self) -> C
    where
        C: FeedRef<Self::Item> + Default
    {
        let mut c = C::default();
        while self.next_and(|x|c.feed_ref(x)).is_some() {}
        c
    }

    fn flat_map<I, F>(self, f: F) -> FlatMapRef<Self, I, F>
    where
        F: FnMut(&Self::Item) -> I,
    {
        FlatMapRef {
            iter: self,
            f,
        }
    }

    fn flat_map1<U, F>(self, f: F) -> FlatMapRef1<Self, U, F>
    where
        U: IntoIterator,
        F: FnMut(&Self::Item) -> U,
    {
        FlatMapRef1 {
            iter: self,
            f,
        }
    }
}

impl<I:Iterator> RefIterator for IterWrapper<I> {
    type Item = I::Item;

    fn next_and<T,F>(&mut self, f: F) -> Option<T>
    where
        F: FnOnce(&Self::Item) -> T
    {
        self.0.next().map(|x|f(&x))
    }
}

impl<I:Iterator,D> RefIterator for IterWrapper2<I,D>
where I::Item: AsRef<D>,
{
    type Item = I::Item;

    fn next_and<T,F>(&mut self, f: F) -> Option<T>
    where
        F: FnOnce(&Self::Item) -> T
    {
        self.0.next().map(|x|f(&x))
    }
}



pub struct FlatMapRef1<I,U,F>
where
    I: RefIterator,
    U: IntoIterator,
    F: FnMut(&I::Item) -> U,
{
    iter: I,
    f: F,
}

impl<I,U,F> RefIterator for FlatMapRef1<I,U,F>
where
    I: RefIterator,
    U: IntoIterator,
    F: for<'a> Fn(&'a I::Item) -> U,
    U::Item: Deref,
{
    type Item = <U::Item as Deref>::Target;

    fn next_and<T,G>(&mut self, g: G) -> Option<T>
    where
        G: FnOnce(&Self::Item) -> T
    {
        loop {
            if let Some(x) = self.iter.next_and(|x|{
                let u = (self.f)(x);
                u.into_iter().next()
            }) {
                return x.map(|x|g(&x));
            }
        }
    }
}


pub struct FlatMapRef<I,U,F>
where
    I: RefIterator,
    U: RefIterator,
    F: FnMut(&I::Item) -> U,
{
    iter: I,
    f: F,
}

impl<I,U,F> RefIterator for FlatMapRef<I,U,F>
where
    I: RefIterator,
    U: RefIterator,
    F: for<'a> Fn(&'a I::Item) -> U,
{
    type Item = U::Item;

    fn next_and<T,G>(&mut self, g: G) -> Option<T>
    where
        G: FnOnce(&Self::Item) -> T
    {
        loop {
            if let Some(x) = self.iter.next_and(|x|{
                let mut u = (self.f)(x);
                u.next_and(g)
            }) {
                return x;
            }
        }
    }
}
pub trait FeedRef<T:?Sized> {
    fn feed_ref(&mut self, item: &T);
}

// pub trait FromIterWrapper<T> {
//     fn from_iter_wrapper<I:Iterator<Item=T>>(iter: IterWrapper<I>) -> Self;
// }

// impl<T,C> FromIterWrapper<T> for C
// where
//     C: Default + FeedRef<T>
// {
//     fn from_iter_wrapper<I:Iterator<Item=T>>(iter: IterWrapper<I>) -> Self {
//         let mut c = C::default();
//         for item in iter.0 {
//             c.feed_ref(&item);
//         }
//         c
//     }
// }
