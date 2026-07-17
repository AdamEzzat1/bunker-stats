// src/kernels/resampling/mod.rs

pub mod bootstrap;
pub mod jackknife;

/// Serial fallback for rayon's parallel iterators when the `parallel` feature
/// is off.
///
/// The resampling kernels call `range.into_par_iter()` followed by `.map(..)`,
/// `.map_init(init, ..)`, `.collect()`, and `.sum()`. When `parallel` is
/// disabled, rayon is not linked, so this shim reproduces exactly those methods
/// with single-threaded semantics. Because the rayon prelude and this trait are
/// never imported at the same time (their `use` statements are
/// `#[cfg]`-mutually-exclusive), there is never any ambiguity about which
/// `into_par_iter` is in scope.
#[cfg(not(feature = "parallel"))]
pub(crate) use serial::IntoParIterCompat;

#[cfg(not(feature = "parallel"))]
mod serial {
    /// Wraps a standard iterator and offers the subset of rayon's
    /// `ParallelIterator` API the resampling kernels use.
    pub(crate) struct SerialIter<I>(pub(crate) I);

    impl<I: Iterator> Iterator for SerialIter<I> {
        type Item = I::Item;
        #[inline]
        fn next(&mut self) -> Option<I::Item> {
            self.0.next()
        }
    }

    impl<I: Iterator> SerialIter<I> {
        /// Serial analogue of `rayon`'s `map_init`: build the state once, then
        /// pass `&mut state` to the closure for every item (rayon would build
        /// one state per worker thread; with a single thread that is one state).
        #[inline]
        pub(crate) fn map_init<T, B, INIT, F>(
            self,
            init: INIT,
            mut f: F,
        ) -> SerialIter<impl Iterator<Item = B>>
        where
            INIT: FnOnce() -> T,
            F: FnMut(&mut T, I::Item) -> B,
        {
            let mut state = init();
            SerialIter(self.0.map(move |item| f(&mut state, item)))
        }
    }

    pub(crate) trait IntoParIterCompat: IntoIterator + Sized {
        #[inline]
        fn into_par_iter(self) -> SerialIter<<Self as IntoIterator>::IntoIter> {
            SerialIter(self.into_iter())
        }
    }

    impl<T: IntoIterator> IntoParIterCompat for T {}
}
