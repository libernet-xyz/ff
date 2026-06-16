# Finite Field Infrastructure

[![CI](https://img.shields.io/github/actions/workflow/status/libernet-xyz/ff/ci.yml?label=CI)](https://github.com/libernet-xyz/ff/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/starkom-ff)](https://crates.io/crates/starkom-ff)
[![license](https://img.shields.io/crates/l/starkom-ff)](https://github.com/libernet-xyz/ff/blob/main/LICENSE)

# Overview

This crate provides the necessary infrastructure to work with finite fields for cryptographic
applications, including prime fields.

The crate is freely inspired by [`ff`](https://crates.io/crates/ff) with only a few minor
improvements, and it's published as [`starkom-ff`](https://crates.io/crates/starkom-ff).

For reference and testing purposes, this crate also provides an implementation of the BLS12-381
scalar field (order `0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001`) based on
our own `Field256` and `PrimeField256` traits.
