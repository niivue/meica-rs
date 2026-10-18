//! Elementary functions with identical bits on every target.
//!
//! `exp`, `expm1`, `tanh` and the sin/cos kernels are ports of FreeBSD msun (fdlibm, Sun
//! Microsystems, freely redistributable with its notice; see NOTICE); `ln` keeps msun's
//! reduction with a double-double series. Only IEEE +, -, *, / and sqrt are used, so native
//! and WASM builds agree bit for bit.

// Constants are copied digit for digit from msun so they can be diffed against it.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-01;
const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
const INV_LN2: f64 = 1.442_695_040_888_963_387_00e+00;
const TWO1023: f64 = 8.988_465_674_311_579_538_65e+307;

fn hi(x: f64) -> i32 {
    (x.to_bits() >> 32) as i32
}

fn with_hi(x: f64, h: i32) -> f64 {
    f64::from_bits(((h as u32 as u64) << 32) | (x.to_bits() & 0xffff_ffff))
}

/// `y * 2^k` for a `y` near 1 and `k` in the normal-exponent range.
fn scale(y: f64, k: i32) -> f64 {
    f64::from_bits(y.to_bits().wrapping_add((k as i64 as u64) << 52))
}

/// msun e_exp.c.
pub fn exp(x: f64) -> f64 {
    const P1: f64 = 1.666_666_666_666_660_190_37e-01;
    const P2: f64 = -2.777_777_777_701_559_338_42e-03;
    const P3: f64 = 6.613_756_321_437_934_361_17e-05;
    const P4: f64 = -1.653_390_220_546_525_153_90e-06;
    const P5: f64 = 4.138_136_797_057_238_460_39e-08;
    let neg = x.is_sign_negative();
    let hx = hi(x) & 0x7fff_ffff;
    if hx >= 0x4086_2e42 {
        if hx >= 0x7ff0_0000 {
            return if x.is_nan() || !neg { x + x } else { 0.0 };
        }
        if x > 7.097_827_128_933_839_730_96e+02 {
            return f64::INFINITY;
        }
        if x < -7.451_332_191_019_411_084_20e+02 {
            return 0.0;
        }
    }
    let mut x = x;
    let (hi_part, lo_part, k);
    if hx > 0x3fd6_2e42 {
        if hx < 0x3ff0_a2b2 {
            let s = if neg { -1.0 } else { 1.0 };
            hi_part = x - s * LN2_HI;
            lo_part = s * LN2_LO;
            k = if neg { -1 } else { 1 };
        } else {
            k = (INV_LN2 * x + if neg { -0.5 } else { 0.5 }) as i32;
            let t = k as f64;
            hi_part = x - t * LN2_HI;
            lo_part = t * LN2_LO;
        }
        x = hi_part - lo_part;
    } else if hx < 0x3e30_0000 {
        return 1.0 + x;
    } else {
        (hi_part, lo_part, k) = (0.0, 0.0, 0);
    }
    let t = x * x;
    let c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if k == 0 {
        return 1.0 - ((x * c) / (c - 2.0) - x);
    }
    let y = 1.0 - ((lo_part - (x * c) / (2.0 - c)) - hi_part);
    if k >= -1021 {
        if k == 1024 {
            return y * 2.0 * TWO1023;
        }
        return scale(y, k);
    }
    scale(y, k + 1000) * 9.332_636_185_032_188_789_90e-302
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Dekker's exact product a*b = p + e, without FMA.
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let split = |v: f64| {
        let c = 134_217_729.0 * v;
        let h = c - (c - v);
        (h, v - h)
    };
    let p = a * b;
    let ((ah, al), (bh, bl)) = (split(a), split(b));
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

/// Natural logarithm, correctly rounded except within ~2^-12 ulp of a halfway case.
/// msun's reduction x = 2^k (1 + f), then log(1 + f) = 2 atanh(s), s = f / (2 + f), carried in
/// double-double through the s^3 term. numpy's macOS log is correctly rounded almost everywhere,
/// and FastICA's w_init (via legacy gauss) needs its exact bits.
pub fn ln(x: f64) -> f64 {
    let mut x = x;
    let mut hx = hi(x);
    let mut k = 0;
    if hx < 0x0010_0000 {
        if x == 0.0 {
            return f64::NEG_INFINITY;
        }
        if hx < 0 {
            return f64::NAN;
        }
        k -= 54;
        x *= 1.801_439_850_948_198_400_00e+16;
        hx = hi(x);
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000f_ffff;
    let i = (hx + 0x95f64) & 0x10_0000;
    x = with_hi(x, hx | (i ^ 0x3ff0_0000));
    k += i >> 20;
    let f = x - 1.0; // exact: x in [sqrt(2)/2, sqrt(2))
    let dk = k as f64;
    if f == 0.0 {
        return dk * LN2_HI + dk * LN2_LO;
    }
    let (dh, dl) = two_sum(2.0, f);
    let sh = f / dh;
    let (p, pe) = two_prod(sh, dh);
    let sl = (((f - p) - pe) - sh * dl) / dh;
    let (q, qe) = two_prod(sh, sh);
    // s^3 = c + c_lo
    let (c, ce) = two_prod(sh, q);
    let c_lo = ce + sh * qe + 3.0 * q * sl;
    // (2/3) s^3 = m + m_lo
    const TT_HI: f64 = 2.0 / 3.0;
    const TT_LO: f64 = 3.700_743_415_417_188e-17;
    let (m, me) = two_prod(TT_HI, c);
    let m_lo = me + TT_HI * c_lo + TT_LO * c;
    // (2/5) s^5 = n + n_lo, from s^5 = s^3 (q + qe + 2 sh sl)
    const TF_HI: f64 = 0.4;
    const TF_LO: f64 = -2.220_446_049_250_313_2e-17;
    let (f5, f5e) = two_prod(c, q);
    let f5_lo = f5e + c_lo * q + c * (qe + 2.0 * sh * sl);
    let (n, ne) = two_prod(TF_HI, f5);
    let n_lo = ne + TF_HI * f5_lo + TF_LO * f5;
    // 2 s^7 (1/7 + z/9 + ... + z^9/25), z = s^2: < 1e-5 of the total, so plain f64 suffices.
    let tail = 2.0
        * f5
        * q
        * (2..=11)
            .rev()
            .fold(0.0, |acc, j| acc * q + 1.0 / (2 * j + 3) as f64);
    let (h, e1) = two_sum(dk * LN2_HI, 2.0 * sh);
    let (h, e2) = two_sum(h, m);
    let (h, e3) = two_sum(h, n);
    h + (e1 + e2 + e3 + 2.0 * sl + m_lo + n_lo + tail + dk * LN2_LO)
}

/// float32 log as numpy computes it on f32 arrays: the correctly rounded f64 log, rounded to f32
/// (double rounding can differ from a correctly rounded logf with probability ~2^-29).
pub fn ln_f32(x: f32) -> f32 {
    ln(x as f64) as f32
}

/// `x^y` for x > 0 as exp(y ln x): within a few ulp for |y ln x| up to ~10, not correctly
/// rounded (numpy calls libm pow).
pub fn pow(x: f64, y: f64) -> f64 {
    exp(y * ln(x))
}

/// `ln(1 + y)` to a few ulp (Goldberg's correction; exact when `1 + y` rounds to 1).
pub fn ln1p(y: f64) -> f64 {
    let u = 1.0 + y;
    if u == 1.0 { y } else { ln(u) * (y / (u - 1.0)) }
}

/// msun s_expm1.c.
pub fn expm1(x: f64) -> f64 {
    const Q1: f64 = -3.333_333_333_333_313_164_28e-02;
    const Q2: f64 = 1.587_301_587_254_814_601_65e-03;
    const Q3: f64 = -7.936_507_578_674_879_424_73e-05;
    const Q4: f64 = 4.008_217_827_329_362_395_52e-06;
    const Q5: f64 = -2.010_992_181_836_243_713_26e-07;
    let neg = x.is_sign_negative();
    let hx = hi(x) & 0x7fff_ffff;
    if hx >= 0x4043_687a {
        if hx >= 0x4086_2e42 {
            if hx >= 0x7ff0_0000 {
                return if x.is_nan() || !neg { x + x } else { -1.0 };
            }
            if x > 7.097_827_128_933_839_730_96e+02 {
                return f64::INFINITY;
            }
        }
        if neg {
            return -1.0;
        }
    }
    let mut x = x;
    let (k, c);
    if hx > 0x3fd6_2e42 {
        let (h, l);
        if hx < 0x3ff0_a2b2 {
            (h, l, k) = if neg {
                (x + LN2_HI, -LN2_LO, -1)
            } else {
                (x - LN2_HI, LN2_LO, 1)
            };
        } else {
            k = (INV_LN2 * x + if neg { -0.5 } else { 0.5 }) as i32;
            let t = k as f64;
            (h, l) = (x - t * LN2_HI, t * LN2_LO);
        }
        x = h - l;
        c = (h - x) - l;
    } else if hx < 0x3c90_0000 {
        return x;
    } else {
        (k, c) = (0, 0.0);
    }
    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let r1 = 1.0 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
    let t = 3.0 - r1 * hfx;
    let mut e = hxs * ((r1 - t) / (6.0 - x * t));
    if k == 0 {
        return x - (x * e - hxs);
    }
    e = x * (e - c) - c;
    e -= hxs;
    if k == -1 {
        return 0.5 * (x - e) - 0.5;
    }
    if k == 1 {
        return if x < -0.25 {
            -2.0 * (e - (x + 0.5))
        } else {
            1.0 + 2.0 * (x - e)
        };
    }
    if k <= -2 || k > 56 {
        let y = 1.0 - (e - x);
        let y = if k == 1024 {
            y * 2.0 * TWO1023
        } else {
            scale(y, k)
        };
        return y - 1.0;
    }
    if k < 20 {
        let t = f64::from_bits(((0x3ff0_0000 - (0x20_0000 >> k)) as u64) << 32);
        scale(t - (e - x), k)
    } else {
        let t = f64::from_bits(((0x3ff - k) as u64) << 52);
        scale((x - (e + t)) + 1.0, k)
    }
}

/// msun s_tanh.c.
pub fn tanh(x: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        return if x.is_nan() { x + x } else { x.signum() };
    }
    let z = if ix < 0x4036_0000 {
        if ix < 0x3e30_0000 {
            return x;
        }
        let a = x.abs();
        if ix >= 0x3ff0_0000 {
            1.0 - 2.0 / (expm1(2.0 * a) + 2.0)
        } else {
            let t = expm1(-2.0 * a);
            -t / (t + 2.0)
        }
    } else {
        1.0
    };
    if x.is_sign_negative() { -z } else { z }
}

/// msun k_sin.c / k_cos.c, valid for |x| <= pi/4.
fn kernel_sin(x: f64) -> f64 {
    const S1: f64 = -1.666_666_666_666_663_243_48e-01;
    const S2: f64 = 8.333_333_333_322_489_461_24e-03;
    const S3: f64 = -1.984_126_982_985_794_931_34e-04;
    const S4: f64 = 2.755_731_370_707_006_767_89e-06;
    const S5: f64 = -2.505_076_025_340_686_341_95e-08;
    const S6: f64 = 1.589_690_995_211_550_102_21e-10;
    let z = x * x;
    let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    x + z * x * (S1 + z * r)
}

fn kernel_cos(x: f64) -> f64 {
    const C1: f64 = 4.166_666_666_666_660_190_37e-02;
    const C2: f64 = -1.388_888_888_887_410_957_49e-03;
    const C3: f64 = 2.480_158_728_947_672_941_78e-05;
    const C4: f64 = -2.755_731_435_139_066_330_35e-07;
    const C5: f64 = 2.087_572_321_298_174_827_90e-09;
    const C6: f64 = -1.135_964_755_778_819_482_65e-11;
    let z = x * x;
    let w = z * z;
    let r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    w + (((1.0 - w) - hz) + z * r)
}

/// `(sin, cos)` of `2*pi*k/n`, reduced exactly by octant in integers (FFT twiddles).
pub fn sincos_2pi(k: u64, n: u64) -> (f64, f64) {
    let a = 8 * (k % n);
    let oct = a / n;
    let rem = a - oct * n;
    let frac = if oct % 2 == 0 { rem } else { n - rem };
    let t = std::f64::consts::FRAC_PI_4 * (frac as f64 / n as f64);
    let (s, c) = (kernel_sin(t), kernel_cos(t));
    match oct {
        0 => (s, c),
        1 => (c, s),
        2 => (c, -s),
        3 => (s, -c),
        4 => (-s, -c),
        5 => (-c, -s),
        6 => (-c, s),
        _ => (-s, c),
    }
}

/// `lgamma(x) - stirling(x)` for x >= 8 (asymptotic series, error < 1e-16).
fn stirling_tail(x: f64) -> f64 {
    let z = 1.0 / (x * x);
    (1.0 / 12.0
        - z * (1.0 / 360.0
            - z * (1.0 / 1260.0
                - z * (1.0 / 1680.0
                    - z * (1.0 / 1188.0
                        - z * (691.0 / 360_360.0 - z * (1.0 / 156.0 - z * 3617.0 / 122_400.0)))))))
        / x
}

const HALF_LN_2PI: f64 = 0.918_938_533_204_672_741_78;

/// ln Gamma(x) for x > 0.
pub fn lgamma(x: f64) -> f64 {
    let mut x = x;
    let mut prod = 1.0;
    while x < 8.0 {
        prod *= x;
        x += 1.0;
    }
    (x - 0.5) * ln(x) - x + HALF_LN_2PI + stirling_tail(x) - ln(prod)
}

/// ln Beta(a, b) for a, b > 0, without the cancellation of three large lgammas.
pub fn lbeta(a: f64, b: f64) -> f64 {
    let (a, b) = if a >= b { (a, b) } else { (b, a) };
    if a < 8.0 {
        return lgamma(a) + lgamma(b) - lgamma(a + b);
    }
    // lgamma(a) - lgamma(a + b) from the Stirling forms, with the large logs folded into ln1p.
    let diff =
        b - (a - 0.5) * ln1p(b / a) - b * ln(a + b) + stirling_tail(a) - stirling_tail(a + b);
    lgamma(b) + diff
}
