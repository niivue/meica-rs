//! numpy's legacy `RandomState` (MT19937), reproduced draw for draw so that a seed gives
//! sklearn FastICA the same `w_init` as Python.

use crate::math::ln;

const N: usize = 624;

pub struct RandomState {
    pub key: [u32; N],
    pos: usize,
    gauss: Option<f64>,
}

impl RandomState {
    /// numpy `_legacy_seeding` for an integer seed (`mt19937_seed`).
    pub fn new(seed: u32) -> Self {
        let mut key = [0u32; N];
        key[0] = seed;
        for i in 1..N {
            key[i] = 1_812_433_253u32
                .wrapping_mul(key[i - 1] ^ (key[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self {
            key,
            pos: N,
            gauss: None,
        }
    }

    fn reload(&mut self) {
        let k = &mut self.key;
        let mix = |a: u32, b: u32| (a & 0x8000_0000) | (b & 0x7fff_ffff);
        let twist = |y: u32| (y >> 1) ^ if y & 1 == 1 { 0x9908_b0df } else { 0 };
        for i in 0..N {
            k[i] = k[(i + 397) % N] ^ twist(mix(k[i], k[(i + 1) % N]));
        }
        self.pos = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.pos == N {
            self.reload();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random_sample`: 53-bit double in [0, 1).
    pub fn random_sample(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// `legacy_gauss`: Marsaglia polar method, second value cached.
    pub fn normal(&mut self) -> f64 {
        if let Some(g) = self.gauss.take() {
            return g;
        }
        let (x1, x2, r2) = loop {
            let x1 = 2.0 * self.random_sample() - 1.0;
            let x2 = 2.0 * self.random_sample() - 1.0;
            // numpy's macOS arm64 build contracts x1*x1 + x2*x2 to this FMA; near r2 = 1 the
            // unfused sum moves ln(r2) by many ulp. f64::mul_add is exactly rounded on every target.
            let r2 = x1.mul_add(x1, x2 * x2);
            if r2 < 1.0 && r2 != 0.0 {
                break (x1, x2, r2);
            }
        };
        let f = (-2.0 * ln(r2) / r2).sqrt();
        self.gauss = Some(f * x1);
        f * x2
    }
}

/// numpy `default_rng(seed)`: SeedSequence-seeded PCG64 (XSL-RR 128/64), with the Generator
/// draws tedana's max_rp_corr uses. Draw for draw as numpy 2.3.
pub struct Pcg64 {
    state: u128,
    inc: u128,
    half: Option<u32>,
}

impl Pcg64 {
    const MULT: u128 = (2_549_297_995_355_413_924u128 << 64) | 4_865_540_595_714_422_341;

    pub fn new(seed: u64) -> Self {
        // SeedSequence(seed): entropy as 32-bit words (one zero word for 0), a 4-word pool.
        let mut entropy = Vec::new();
        let mut n = seed;
        while n > 0 || entropy.is_empty() {
            entropy.push(n as u32);
            n >>= 32;
        }
        let mut hash = 0x43b0_d7e5u32;
        let mut hashmix = |v: u32| {
            let mut v = v ^ hash;
            hash = hash.wrapping_mul(0x931e_8875);
            v = v.wrapping_mul(hash);
            v ^ (v >> 16)
        };
        let mix = |x: u32, y: u32| {
            let r = 0xca01_f9ddu32
                .wrapping_mul(x)
                .wrapping_sub(0x4973_f715u32.wrapping_mul(y));
            r ^ (r >> 16)
        };
        let mut pool = [0u32; 4];
        for (i, p) in pool.iter_mut().enumerate() {
            *p = hashmix(entropy.get(i).copied().unwrap_or(0));
        }
        for src in 0..4 {
            for dst in 0..4 {
                if src != dst {
                    let h = hashmix(pool[src]);
                    pool[dst] = mix(pool[dst], h);
                }
            }
        }
        for &e in entropy.iter().skip(4) {
            for p in pool.iter_mut() {
                let h = hashmix(e);
                *p = mix(*p, h);
            }
        }
        // generate_state(4, uint64): 8 words cycling the pool, paired little-endian.
        let mut hash = 0x8b51_f9ddu32;
        let words: Vec<u64> = (0..8)
            .map(|i| {
                let mut v = pool[i % 4] ^ hash;
                hash = hash.wrapping_mul(0x58f3_8ded);
                v = v.wrapping_mul(hash);
                (v ^ (v >> 16)) as u64
            })
            .collect::<Vec<_>>()
            .chunks(2)
            .map(|w| w[0] | (w[1] << 32))
            .collect();
        let initstate = ((words[0] as u128) << 64) | words[1] as u128;
        let initseq = ((words[2] as u128) << 64) | words[3] as u128;
        let mut g = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
            half: None,
        };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(Self::MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        ((s >> 64) as u64 ^ s as u64).rotate_right((s >> 122) as u32)
    }

    /// Low half of a 64-bit draw, then its high half.
    fn next_u32(&mut self) -> u32 {
        if let Some(h) = self.half.take() {
            return h;
        }
        let v = self.next_u64();
        self.half = Some((v >> 32) as u32);
        v as u32
    }

    /// `random_bounded_uint64(0, rng, use_masked=False)` for rng < 2^32 - 1: Lemire's method on
    /// 32-bit draws, uniform on 0..=rng.
    fn bounded(&mut self, rng: u32) -> usize {
        if rng == 0 {
            return 0;
        }
        let excl = rng + 1;
        let mut m = self.next_u32() as u64 * excl as u64;
        if (m as u32) < excl {
            let threshold = (u32::MAX - rng) % excl;
            while (m as u32) < threshold {
                m = self.next_u32() as u64 * excl as u64;
            }
        }
        (m >> 32) as usize
    }

    /// `Generator.choice(n, k, replace=False)`: Floyd's sampling then a shuffle, or for n > 10000
    /// with k > n/50 a partial tail shuffle (only the Floyd branch is fixture-checked).
    pub fn choice(&mut self, n: usize, k: usize) -> Vec<usize> {
        if n > 10_000 && k > n / 50 {
            let mut idx: Vec<usize> = (0..n).collect();
            for i in ((n - k).max(1)..n).rev() {
                let j = self.bounded(i as u32);
                idx.swap(i, j);
            }
            return idx.split_off(n - k);
        }
        let mut out: Vec<usize> = Vec::with_capacity(k);
        let mut seen = std::collections::HashSet::with_capacity(k);
        for j in n - k..n {
            let v = self.bounded(j as u32);
            let pick = if seen.insert(v) { v } else { j };
            seen.insert(pick);
            out.push(pick);
        }
        for i in (1..k).rev() {
            let j = self.bounded(i as u32);
            out.swap(i, j);
        }
        out
    }
}
