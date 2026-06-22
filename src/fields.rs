use primitive_types::{H512, U256, U512};
use rand_core::{CryptoRng, TryCryptoRng};
use std::fmt::{Binary, Debug, Display, LowerHex, Octal, UpperHex};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::str::FromStr;
use subtle::{
    Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
    CtOption,
};

/// A finite field.
///
/// NOTE: this is assumed to be a *numeric* finite field, ie. a range of the integers from 0
/// inclusive to N exclusive, with N being the cardinality of the field. For this reason this trait
/// inherits several traits that are not necessarily part of algebraic fields, such as [`Ord`] for
/// total ordering and several `std::fmt::*` traits for formatting numbers.
pub trait Field:
    'static
    + Debug
    + Default
    + Sized
    + Send
    + Sync
    + Copy
    + Clone
    + Eq
    + Ord
    + ConstantTimeEq
    + ConstantTimeGreater
    + ConstantTimeLess
    + ConditionallySelectable
    + Add<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + AddAssign<Self>
    + for<'a> AddAssign<&'a Self>
    + Neg<Output = Self>
    + Sub<Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + SubAssign<Self>
    + for<'a> SubAssign<&'a Self>
    + Mul<Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + MulAssign<Self>
    + for<'a> MulAssign<&'a Self>
    + Div<Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + DivAssign<Self>
    + for<'a> DivAssign<&'a Self>
    + Sum
    + Product
    + Display
    + Binary
    + Octal
    + LowerHex
    + UpperHex
    + FromStr
    + From<u8>
    + From<u16>
{
    /// The number of bytes required to represent a value.
    const LEN: usize;

    /// The number of bits required to represent a value.
    const NUM_BITS: usize = Self::LEN * 8;

    /// Same as [`Self::NUM_BITS`]. Provided for consistency with native Rust types.
    const BITS: usize = Self::NUM_BITS;

    /// The additive identity element.
    const ZERO: Self;

    /// The multiplicative identity element.
    const ONE: Self;

    /// The largest value in the field.
    const MAX: Self;

    /// Returns the element zero.
    fn zero() -> Self {
        Self::ZERO
    }

    /// Returns the element one.
    fn one() -> Self {
        Self::ONE
    }

    /// Compares with zero.
    fn is_zero(&self) -> Choice {
        self.ct_eq(&Self::ZERO)
    }

    /// Returns true iff the value is even.
    fn is_even(&self) -> Choice {
        !self.is_odd()
    }

    /// Returns true iff the value is odd.
    fn is_odd(&self) -> Choice;

    /// Picks a uniformly distributed random scalar securely from the provided fallible CSPRNG.
    fn try_random<R: TryCryptoRng>(rng: &mut R) -> Result<Self, R::Error>;

    /// Picks a uniformly distributed random scalar securely from the provided infallible CSPRNG.
    fn random<R: CryptoRng>(rng: &mut R) -> Self;

    /// Picks a uniformly distributed random scalar securely from the system's default CSPRNG.
    fn random_default() -> Self;

    /// Returns this value doubled. `self` remains unchanged.
    fn double(&self) -> Self {
        self.add(self)
    }

    /// Returns this value squared. `self` remains unchanged.
    fn square(&self) -> Self {
        self.mul(self)
    }

    /// Returns this value raised to 3. `self` remains unchanged.
    fn cube(&self) -> Self {
        self.square() * self
    }

    /// Returns the modular inverse of `self, or `None` if `self` is zero.
    fn invert(&self) -> CtOption<Self>;

    /// Returns the modular inverse of `self`, assuming `self` is not zero and panicking otherwise.
    fn invert_unwrap(&self) -> Self {
        self.invert().unwrap()
    }

    /// Returns the modular inverse of `self`, or zero if `self` is zero.
    fn invert_or_zero(&self) -> Self {
        self.invert().unwrap_or(Self::ZERO)
    }

    /// Returns the modular inverse of `self, or `None` if `self` is zero.
    fn invert_vartime(&self) -> Option<Self>;

    /// Raises this value to `exp`, running exactly [`Self::NUM_BITS`] squares and multiplications
    /// so that a time observer cannot infer the exponent.
    fn pow(self, exp: Self) -> Self;

    /// Raises this value to `exp`.
    fn pow_vartime(self, exp: Self) -> Self;

    /// Raises this value to `exp`, running exactly [`usize::BITS`] squares and multiplications so
    /// that a time observer cannot infer the exponent.
    ///
    /// Unlike [`Field::pow`], `exp` is a `usize`. That makes the algorithm significantly faster
    /// because the square-and-multiply loop runs only [`usize::BITS`] times rather than
    /// [`Field::BITS`] times, and bitwise operations on the exponent are native.
    fn pow_small(mut self, mut exp: usize) -> Self {
        let mut result = Self::ONE;
        for _ in 0..usize::BITS {
            let product = result * self;
            result = Self::conditional_select(&result, &product, Choice::from((exp & 1) as u8));
            exp >>= 1;
            self = self.square();
        }
        result
    }

    /// Raises this value to `exp`.
    ///
    /// Unlike [`Field::pow`], `exp` is a `usize`. That makes the algorithm significantly faster
    /// because bitwise operations on the exponent are native.
    fn pow_small_vartime(mut self, mut exp: usize) -> Self {
        let mut result = Self::ONE;
        while exp != 0 {
            if (exp & 1) != 0 {
                result *= self;
            }
            exp >>= 1;
            self = self.square();
        }
        result
    }

    /// Raises this value to `exp`, running exactly [`u32::BITS`] squares and multiplications so
    /// that an observer cannot infer the exponent.
    ///
    /// Unlike [`Field::pow`], `exp` is a `u32`. That makes the algorithm significantly faster
    /// because the square-and-multiply loop runs only 32 times rather than [`Field::BITS`] times,
    /// and bitwise operations on the exponent are native.
    fn pow_u32(mut self, mut exp: u32) -> Self {
        let mut result = Self::ONE;
        for _ in 0..u32::BITS {
            let product = result * self;
            result = Self::conditional_select(&result, &product, Choice::from((exp & 1) as u8));
            exp >>= 1;
            self = self.square();
        }
        result
    }

    /// Raises this value to `exp`.
    ///
    /// Unlike [`Field::pow_vartime`], `exp` is a `u32`. That makes the algorithm significantly
    /// faster because bitwise operations on the exponent are native.
    fn pow_u32_vartime(mut self, mut exp: u32) -> Self {
        let mut result = Self::ONE;
        while exp != 0 {
            if (exp & 1) != 0 {
                result *= self;
            }
            exp >>= 1;
            self = self.square();
        }
        result
    }

    /// Raises this value to `exp`, running exactly [`u64::BITS`] squares and multiplications so
    /// that an observer cannot infer the exponent.
    ///
    /// Unlike [`Field::pow`], `exp` is a `u64`. That makes the algorithm significantly faster
    /// because the square-and-multiply loop runs only 64 times rather than [`Field::BITS`] times,
    /// and bitwise operations on the exponent are native.
    fn pow_u64(mut self, mut exp: u64) -> Self {
        let mut result = Self::ONE;
        for _ in 0..u64::BITS {
            let product = result * self;
            result = Self::conditional_select(&result, &product, Choice::from((exp & 1) as u8));
            exp >>= 1;
            self = self.square();
        }
        result
    }

    /// Raises this value to `exp`.
    ///
    /// Unlike [`Field::pow_vartime`], `exp` is a `u64`. That makes the algorithm significantly
    /// faster because bitwise operations on the exponent are native.
    fn pow_u64_vartime(mut self, mut exp: u64) -> Self {
        let mut result = Self::ONE;
        while exp != 0 {
            if (exp & 1) != 0 {
                result *= self;
            }
            exp >>= 1;
            self = self.square();
        }
        result
    }

    /// Performs integer division by `rhs` and returns a (quotient, remainder) pair. Panics if `rhs`
    /// is zero.
    fn div_int(&self, rhs: &Self) -> (Self, Self);

    /// Constructs a scalar from the little-endian byte representation of an integer.
    ///
    /// The provided slice must have exactly [`Self::LEN`] bytes.
    ///
    /// The function returns `None` if the integer lies outside the field range.
    fn try_from_le_bytes(bytes: &[u8]) -> CtOption<Self>;

    /// Constructs a scalar from the big-endian byte representation of an integer.
    ///
    /// The provided slice must have exactly [`Self::LEN`] bytes.
    ///
    /// The function returns `None` if the integer lies outside the field range.
    fn try_from_be_bytes(bytes: &[u8]) -> CtOption<Self>;

    /// Parses a scalar from its text representation in the given `radix`.
    ///
    /// Returns an error on invalid format or overflow. Panics if `radix` is less than 2 or greater
    /// than 36.
    fn from_str_radix(s: &str, radix: usize) -> Result<Self, std::fmt::Error>;

    /// Converts a scalar to its textual representation in the given `radix`.
    ///
    /// The returned string will have at least `pad_to` characters, and will be padded with zeros if
    /// necessary.
    ///
    /// When `radix` is greater than 10, the representation will start using alphabetic characters
    /// from A to Z for digits greater than 9, e.g. character A to F for hexadecimal numbers. The
    /// `upper_case` flag specifies whether those characters must be lower case or upper case.
    ///
    /// This function panics if `radix` is less than 2 or greater than 36.
    fn to_str_radix(&self, radix: usize, pad_to: usize, upper_case: bool) -> String;

    /// Returns this scalar as a `u8`, or `None` if the value exceeds the 8-bit range.
    fn try_to_u8(&self) -> Option<u8>;

    /// Returns this scalar as a `u16`, or `None` if the value exceeds the 16-bit range.
    fn try_to_u16(&self) -> Option<u16>;
}

/// A ~64-bit [`Field`].
pub trait Field64: Field + From<u32> + TryFrom<u64> {
    /// Returns the little-endian representation of the scalar.
    fn to_le_bytes(&self) -> [u8; 8];

    /// Returns the big-endian representation of the scalar.
    fn to_be_bytes(&self) -> [u8; 8];

    /// Constructs a scalar from a 128-bit unsigned value, using modular reduction to fit it into
    /// the scalar range.
    fn from_u128_mod_n(u128: u128) -> Self;

    /// Constructs a scalar from a 256-bit unsigned value, using modular reduction to fit it into
    /// the scalar range.
    fn from_u256_mod_n(u256: U256) -> Self;

    /// Returns this scalar as a `u32`, or `None` if the value exceeds the 32-bit range.
    fn try_to_u32(&self) -> CtOption<u32>;

    /// Converts the scalar to a 64-bit unsigned integer.
    fn to_u64(&self) -> u64;

    /// Converts the scalar to a 128-bit unsigned integer.
    fn to_u128(&self) -> u128;

    /// Converts the scalar to a 256-bit unsigned integer.
    fn to_u256(&self) -> U256;

    /// Converts the scalar to a 512-bit unsigned integer.
    fn to_u512(&self) -> U512;
}

/// A ~256-bit [`Field`].
pub trait Field256: Field + From<u32> + From<u64> + From<u128> + TryFrom<U256> {
    /// Returns the little-endian representation of the scalar.
    fn to_le_bytes(&self) -> [u8; 32];

    /// Returns the big-endian representation of the scalar.
    fn to_be_bytes(&self) -> [u8; 32];

    /// Constructs a scalar from a 512-bit unsigned value, using modular reduction to fit it into
    /// the scalar range.
    fn from_u512_mod_n(u512: U512) -> Self;

    /// Constructs a scalar from an [`H512`].
    ///
    /// This function works by converting the [`H512`] to a [`U512`] using little-endian byte order
    /// and then calling [`Self::from_u512_mod_n`].
    fn from_h512(h512: H512) -> Self;

    /// Returns this scalar as a `u32`, or `None` if the value exceeds the 32-bit range.
    fn try_to_u32(&self) -> CtOption<u32>;

    /// Returns this scalar as a `u64`, or `None` if the value exceeds the 64-bit range.
    fn try_to_u64(&self) -> CtOption<u64>;

    /// Returns this scalar as a `u128`, or `None` if the value exceeds the 128-bit range.
    fn try_to_u128(&self) -> CtOption<u128>;

    /// Returns this scalar as a [`U256`].
    fn to_u256(&self) -> U256;

    /// Returns this scalar as a [`U512`].
    fn to_u512(&self) -> U512;
}

/// A [`Field`] whose order is a prime number.
///
/// This kind of field has certain mathematical properties that are very useful in cryptographic
/// applications. Notably, Fermat's Little Theorem holds.
pub trait PrimeField: Field {
    /// The prime order of the field.
    ///
    /// Must be consistent with the [`Field::MAX`] constant.
    const MODULUS: &'static str;

    /// The 2-adicity of the field, which is the exponent of 2 in the factorization of p-1.
    const S: usize;

    /// A fixed multiplicative generator of `modulus - 1` order. This element must also be a
    /// quadratic nonresidue.
    ///
    /// Implementations of this trait MUST ensure that this is the generator used to derive
    /// [`Self::ROOT_OF_UNITY`].
    const MULTIPLICATIVE_GENERATOR: Self;

    /// `p-2`, which is the exponent used for modular inversion on prime fields as per Fermat's
    /// Little Theorem.
    const MINUS_TWO: Self;

    /// 2^-1
    const TWO_INV: Self;

    /// A primitive root of unity.
    const ROOT_OF_UNITY: Self;

    /// The modular inverse of [`Self::ROOT_OF_UNITY`].
    const ROOT_OF_UNITY_INV: Self;

    /// Generator of the `t-order` multiplicative subgroup.
    ///
    /// It can be calculated by exponentiating [`Self::MULTIPLICATIVE_GENERATOR`] by `2^s`, where
    /// `s` is [`Self::S`].
    const DELTA: Self;
}

/// A ~64-bit prime field.
pub trait PrimeField64: Field64 + PrimeField {}

/// A ~256-bit prime field.
pub trait PrimeField256: Field256 + PrimeField {}
