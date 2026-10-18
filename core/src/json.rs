//! Minimal JSON for decision trees, sidecars and outputs. Accepts Python's NaN/Infinity.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    /// An integer literal, kept apart from floats so that it prints as Python would.
    Int(i64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>), // insertion order, as Python writes it
}

impl From<&str> for Json {
    fn from(s: &str) -> Self {
        Json::Str(s.into())
    }
}

impl From<Option<f64>> for Json {
    fn from(v: Option<f64>) -> Self {
        v.map_or(Json::Null, Json::Num)
    }
}

impl Json {
    pub fn parse(s: &str) -> Result<Json, String> {
        let mut p = Parser {
            b: s.as_bytes(),
            i: 0,
            depth: 0,
        };
        let v = p.value()?;
        p.ws();
        if p.i != p.b.len() {
            return Err(p.err("trailing characters"));
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Json> {
        match self {
            Json::Obj(kv) => kv.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn strings<'a>(items: impl IntoIterator<Item = &'a String>) -> Json {
        Json::Arr(items.into_iter().map(|s| Json::Str(s.clone())).collect())
    }

    /// The string items of an array (none for other values).
    pub fn str_list(&self) -> Vec<String> {
        self.as_arr()
            .unwrap_or_default()
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect()
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            Json::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Python `json.dump(indent=4, sort_keys=True)` layout.
    pub fn pretty(&self) -> String {
        let mut out = String::new();
        self.write_pretty(&mut out, 0);
        out
    }

    fn write_pretty(&self, out: &mut String, depth: usize) {
        let pad = |n: usize| "    ".repeat(n);
        match self {
            Json::Arr(a) if !a.is_empty() => {
                out.push_str("[\n");
                for (i, v) in a.iter().enumerate() {
                    out.push_str(&pad(depth + 1));
                    v.write_pretty(out, depth + 1);
                    out.push_str(if i + 1 < a.len() { ",\n" } else { "\n" });
                }
                out.push_str(&pad(depth));
                out.push(']');
            }
            Json::Obj(kv) if !kv.is_empty() => {
                let mut kv: Vec<&(String, Json)> = kv.iter().collect();
                kv.sort_by(|a, b| a.0.cmp(&b.0));
                out.push_str("{\n");
                for (i, (k, v)) in kv.iter().enumerate() {
                    out.push_str(&pad(depth + 1));
                    out.push_str(&format!("{}: ", Json::Str(k.clone())));
                    v.write_pretty(out, depth + 1);
                    out.push_str(if i + 1 < kv.len() { ",\n" } else { "\n" });
                }
                out.push_str(&pad(depth));
                out.push('}');
            }
            other => out.push_str(&other.to_string()),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }

    pub fn f64s(&self) -> Option<Vec<f64>> {
        self.as_arr()?.iter().map(Json::as_f64).collect()
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn err(&self, what: &str) -> String {
        format!("JSON: {what} at byte {}", self.i)
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        let hit = self.b[self.i..].starts_with(lit.as_bytes());
        if hit {
            self.i += lit.len();
        }
        hit
    }

    /// Nesting is capped so hostile input cannot overflow the stack (about 1 MB in wasm).
    fn value(&mut self) -> Result<Json, String> {
        if self.depth == 256 {
            return Err(self.err("nested too deeply"));
        }
        self.depth += 1;
        let v = self.value_unbounded();
        self.depth -= 1;
        v
    }

    fn value_unbounded(&mut self) -> Result<Json, String> {
        self.ws();
        let Some(&c) = self.b.get(self.i) else {
            return Err(self.err("unexpected end"));
        };
        match c {
            b'{' => {
                self.i += 1;
                let mut kv = Vec::new();
                self.ws();
                if self.eat("}") {
                    return Ok(Json::Obj(kv));
                }
                loop {
                    self.ws();
                    let Json::Str(k) = self.value()? else {
                        return Err(self.err("expected key"));
                    };
                    self.ws();
                    if !self.eat(":") {
                        return Err(self.err("expected ':'"));
                    }
                    kv.push((k, self.value()?));
                    self.ws();
                    if self.eat("}") {
                        return Ok(Json::Obj(kv));
                    }
                    if !self.eat(",") {
                        return Err(self.err("expected ',' or '}'"));
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.eat("]") {
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    if self.eat("]") {
                        return Ok(Json::Arr(a));
                    }
                    if !self.eat(",") {
                        return Err(self.err("expected ',' or ']'"));
                    }
                }
            }
            b'"' => self.string().map(Json::Str),
            _ if self.eat("true") => Ok(Json::Bool(true)),
            _ if self.eat("false") => Ok(Json::Bool(false)),
            _ if self.eat("null") => Ok(Json::Null),
            _ if self.eat("NaN") => Ok(Json::Num(f64::NAN)),
            _ if self.eat("Infinity") => Ok(Json::Num(f64::INFINITY)),
            _ if self.eat("-Infinity") => Ok(Json::Num(f64::NEG_INFINITY)),
            _ => {
                let start = self.i;
                while self.i < self.b.len()
                    && matches!(
                        self.b[self.i],
                        b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
                    )
                {
                    self.i += 1;
                }
                let text = std::str::from_utf8(&self.b[start..self.i])
                    .map_err(|_| self.err("bad value"))?;
                if let Ok(i) = text.parse::<i64>() {
                    return Ok(Json::Int(i));
                }
                text.parse()
                    .map(Json::Num)
                    .map_err(|_| self.err("bad value"))
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' {
                self.i += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.b[start..self.i]).map_err(|_| self.err("bad UTF-8"))?,
            );
            match self.b.get(self.i) {
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    let e = *self
                        .b
                        .get(self.i + 1)
                        .ok_or_else(|| self.err("bad escape"))?;
                    self.i += 2;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xd800..0xdc00).contains(&cp) && self.eat("\\u") {
                                let lo = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&lo) {
                                    return Err(self.err("bad surrogate pair"));
                                }
                                cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                            }
                            out.push(char::from_u32(cp).ok_or_else(|| self.err("bad \\u"))?);
                        }
                        _ => return Err(self.err("bad escape")),
                    }
                }
                _ => return Err(self.err("unterminated string")),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self
            .b
            .get(self.i..self.i + 4)
            .and_then(|h| u32::from_str_radix(std::str::from_utf8(h).ok()?, 16).ok());
        self.i += 4;
        h.ok_or_else(|| self.err("bad \\u"))
    }
}

/// Python `json.dumps` default layout (`, ` and `: ` separators, ASCII only).
impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Json::Null => f.write_str("null"),
            Json::Bool(b) => write!(f, "{b}"),
            Json::Num(v) => f.write_str(&py_float(*v)),
            Json::Int(i) => write!(f, "{i}"),
            Json::Str(s) => {
                f.write_str("\"")?;
                for c in s.chars() {
                    match c {
                        '"' => f.write_str("\\\"")?,
                        '\\' => f.write_str("\\\\")?,
                        '\n' => f.write_str("\\n")?,
                        c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                            // Python's ensure_ascii, with surrogate pairs above the BMP
                            let mut buf = [0u16; 2];
                            for unit in c.encode_utf16(&mut buf) {
                                write!(f, "\\u{unit:04x}")?;
                            }
                        }
                        c => write!(f, "{c}")?,
                    }
                }
                f.write_str("\"")
            }
            Json::Arr(a) => {
                f.write_str("[")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{v}")?;
                }
                f.write_str("]")
            }
            Json::Obj(kv) => {
                f.write_str("{")?;
                for (i, (k, v)) in kv.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}: {v}", Json::Str(k.clone()))?;
                }
                f.write_str("}")
            }
        }
    }
}

/// Python `repr(float)`: shortest round-trip digits, exponent form below 1e-4 and from 1e16 as
/// `1e-05` / `1e+16`.
pub fn py_float(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    let s = format!("{v:?}");
    match s.split_once('e') {
        Some((m, e)) => {
            let (sign, digits) = if let Some(d) = e.strip_prefix('-') {
                ('-', d)
            } else {
                ('+', e)
            };
            format!("{m}e{sign}{digits:0>2}")
        }
        None => s,
    }
}
