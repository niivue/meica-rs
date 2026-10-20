//! tedana's `ComponentSelector`: a decision-tree interpreter over the JSON trees in `trees/`,
//! with the node functions of `selection_nodes.py` the bundled trees use.
//!
//! Component subsets are kept in ascending order. Python builds them with `list(set(...))`;
//! only `kappa_elbow_kundu`, which reads positions as labels, could notice, and every bundled
//! tree calls it on "all".

use crate::json::Json;
use crate::stats::{f_ppf_dfn1, median, rankdata, sum};
use std::collections::BTreeSet;

const TREES: [(&str, &str); 8] = [
    ("tedana_orig", include_str!("../trees/tedana_orig.json")),
    ("meica", include_str!("../trees/meica.json")),
    ("minimal", include_str!("../trees/minimal.json")),
    (
        "demo_external_regressors_single_model",
        include_str!("../trees/demo_external_regressors_single_model.json"),
    ),
    (
        "demo_external_regressors_motion_task_models",
        include_str!("../trees/demo_external_regressors_motion_task_models.json"),
    ),
    (
        "demo_external_aroma",
        include_str!("../trees/demo_external_aroma.json"),
    ),
    // tedana's demo trees with fMRIPrep confound names (also on HF, for tedana's --tree FILE).
    ("motion_f", include_str!("../trees/motion_f.json")),
    ("motion_aroma", include_str!("../trees/motion_aroma.json")),
];

/// A bundled tree's JSON by name (tedana's "kundu" alias included).
pub fn bundled_tree(name: &str) -> Option<&'static str> {
    let name = if name == "kundu" { "tedana_orig" } else { name };
    TREES.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

/// Whether a tree has an external_regressor_config (and so takes --external).
pub fn takes_regressors(tree: &Json) -> bool {
    tree.get("external_regressor_config")
        .is_some_and(|c| *c != Json::Null)
}

/// A bundled tree by name or a tree file's contents.
pub fn load_tree(name_or_json: &str) -> Result<Json, String> {
    let text = bundled_tree(name_or_json).unwrap_or(name_or_json);
    if !text.trim_start().starts_with('{') {
        return Err(format!(
            "decision tree {:?} is not a bundled name or a JSON object",
            name_or_json.chars().take(80).collect::<String>()
        ));
    }
    let mut tree = Json::parse(text).map_err(|e| format!("decision tree: {e}"))?;
    if tree.get("nodes").and_then(Json::as_arr).is_none() {
        return Err("decision tree has no nodes".into());
    }
    if let Json::Obj(kv) = &mut tree
        && !kv.iter().any(|(k, _)| k == "external_regressor_config")
    {
        kv.push(("external_regressor_config".into(), Json::Null));
    }
    Ok(tree)
}

/// Component table: named metric columns (NaN where undefined) plus classifications and tags.
pub struct Table {
    pub columns: Vec<(String, Vec<f64>)>,
    pub classification: Vec<String>,
    pub tags: Vec<Vec<String>>,
}

impl Table {
    pub fn new(columns: Vec<(String, Vec<f64>)>) -> Self {
        let n = columns.first().map_or(0, |c| c.1.len());
        Table {
            columns,
            classification: vec!["unclassified".into(); n],
            tags: vec![Vec::new(); n],
        }
    }

    pub fn len(&self) -> usize {
        self.classification.len()
    }

    pub fn is_empty(&self) -> bool {
        self.classification.is_empty()
    }

    pub fn col(&self, name: &str) -> Option<&[f64]> {
        self.columns
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| &v[..])
    }

    fn set_col(&mut self, name: &str, values: Vec<f64>) {
        upsert(&mut self.columns, name, values);
    }
}

pub struct Selector {
    pub table: Table,
    /// Cross-component metrics in the order they were set.
    pub cross: Vec<(String, Json)>,
    /// "initialized classification", then a "Node i" column per node that classified.
    pub status: Vec<(String, Vec<String>)>,
    /// The tree with each node's `outputs` filled in.
    pub tree: Json,
    node: usize,
}

fn num(v: f64) -> Json {
    Json::Num(v)
}

fn int(v: usize) -> Json {
    Json::Int(v as i64)
}

/// Replace the value under `key`, or append it.
fn upsert<T>(kv: &mut Vec<(String, T)>, key: &str, value: T) {
    match kv.iter_mut().find(|(k, _)| k == key) {
        Some(e) => e.1 = value,
        None => kv.push((key.into(), value)),
    }
}

/// `scipy.stats.scoreatpercentile(v, per)` with fractional interpolation.
fn score_at_percentile(v: &[f64], per: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let idx = per / 100.0 * (s.len() - 1) as f64;
    let i = idx as usize;
    if i as f64 == idx {
        return s[i] * 1.0 / 1.0;
    }
    let (wj, wi) = ((i + 1) as f64 - idx, idx - i as f64);
    (-0.0 + s[i] * wj + s[i + 1] * wi) / (wj + wi)
}

/// tedana `getelbow(..., return_val=True)`: the point farthest from the line joining the ends
/// of the descending-sorted curve.
fn getelbow(v: &[f64]) -> f64 {
    let mut a = v.to_vec();
    a.sort_by(|x, y| y.total_cmp(x));
    let n = a.len();
    let (bx, by) = ((n - 1) as f64, a[n - 1] - a[0]);
    let norm = (bx * bx + by * by).sqrt();
    let (hx, hy) = (bx / norm, by / norm);
    let mut best = (f64::NEG_INFINITY, 0);
    for (k, &ak) in a.iter().enumerate() {
        let (px, py) = (k as f64, ak - a[0]);
        let proj = hx * px + hy * py;
        let (dx, dy) = (px - proj * hx, py - proj * hy);
        let d = (dx * dx + dy * dy).sqrt();
        if d > best.0 {
            best = (d, k);
        }
    }
    a[best.1]
}

/// tedana `generate_decision_table_score`: mean rank, `ascending` ranked low-is-good.
pub fn decision_table_score(ascending: &[&[f64]], descending: &[&[f64]]) -> Vec<f64> {
    let n = ascending
        .iter()
        .chain(descending)
        .next()
        .map_or(0, |m| m.len());
    let mut ranks: Vec<Vec<f64>> = ascending.iter().map(|m| rankdata(m)).collect();
    ranks.extend(
        descending
            .iter()
            .map(|m| rankdata(m).iter().map(|r| n as f64 - r).collect()),
    );
    (0..n)
        .map(|c| ranks.iter().fold(-0.0, |a, r| a + r[c]) / ranks.len() as f64)
        .collect()
}

fn compare(a: f64, op: &str, b: f64) -> Result<bool, String> {
    Ok(match op {
        ">" => a > b,
        ">=" => a >= b,
        "==" => a == b,
        "<=" => a <= b,
        "<" => a < b,
        _ => {
            return Err(format!(
                "{op} is not a binary comparison operator, like > or <"
            ));
        }
    })
}

impl Selector {
    /// Run every node of `tree` on `table` (tedana `automatic_selection`).
    pub fn select(
        tree: Json,
        table: Table,
        n_echos: usize,
        n_vols: usize,
    ) -> Result<Selector, String> {
        let status = vec![(
            "initialized classification".to_string(),
            table.classification.clone(),
        )];
        let mut s = Selector {
            table,
            cross: vec![
                ("n_echos".into(), int(n_echos)),
                ("n_vols".into(), int(n_vols)),
                ("n_independent_echos".into(), Json::Null),
            ],
            status,
            tree,
            node: 0,
        };
        let nodes = s.tree.get("nodes").and_then(Json::as_arr).unwrap().to_vec();
        let (mut done, mut used_metrics) = (Vec::new(), BTreeSet::new());
        for (i, node) in nodes.iter().enumerate() {
            s.node = i;
            let outputs = s.run(node).map_err(|e| format!("Step {i}: {e}"))?;
            if let Some(used) = outputs.get("used_metrics") {
                used_metrics.extend(used.str_list());
            }
            let mut node = node.clone();
            if let Json::Obj(kv) = &mut node {
                kv.push(("outputs".into(), outputs));
            }
            done.push(node);
        }
        if let Json::Obj(kv) = &mut s.tree {
            upsert(kv, "nodes", Json::Arr(done));
            kv.push(("used_metrics".into(), Json::strings(&used_metrics)));
        }
        Ok(s)
    }

    fn cross_value(&self, key: &str) -> Option<&Json> {
        self.cross.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    fn set_cross(&mut self, key: &str, value: Json) {
        upsert(&mut self.cross, key, value);
    }

    /// Set a cross-component metric and report it in the node's outputs.
    fn publish(&mut self, out: &mut Vec<(String, Json)>, key: &str, value: Json) {
        self.set_cross(key, value.clone());
        out.push((key.into(), value));
    }

    /// `metric` of each of `comps`.
    fn values(&self, metric: &str, comps: &[usize]) -> Result<Vec<f64>, String> {
        let col = self.table.col(metric).ok_or(format!("{metric} missing"))?;
        Ok(comps.iter().map(|&c| col[c]).collect())
    }

    fn cross_f64(&self, key: &str) -> Result<f64, String> {
        self.cross_value(key)
            .and_then(Json::as_f64)
            .ok_or_else(|| format!("{key} not in cross_component_metrics"))
    }

    /// The elbow nodes' leading outputs; returns n_independent_echos when set, else n_echos.
    fn elbow_outputs(&self, out: &mut Vec<(String, Json)>, label: (String, Json)) -> f64 {
        let n_echos = self.cross_value("n_echos").cloned().unwrap();
        let n_ind = self
            .cross_value("n_independent_echos")
            .and_then(Json::as_f64)
            .or(n_echos.as_f64())
            .unwrap();
        out.push(label);
        out.push(("n_echos".into(), n_echos));
        out.push(("n_independent_echos".into(), int(n_ind as usize)));
        n_ind
    }

    /// tedana `selectcomps2use`.
    fn comps(&self, decide: &Json) -> Result<Vec<usize>, String> {
        let items: Vec<Json> = match decide {
            Json::Arr(a) => a.clone(),
            other => vec![other.clone()],
        };
        if items.first().and_then(Json::as_str) == Some("all") {
            return Ok((0..self.table.len()).collect());
        }
        if items.iter().all(|i| i.as_str().is_some()) {
            let wanted: Vec<&str> = items.iter().filter_map(Json::as_str).collect();
            return Ok((0..self.table.len())
                .filter(|&c| wanted.contains(&self.table.classification[c].as_str()))
                .collect());
        }
        items
            .iter()
            .map(|i| {
                i.as_f64()
                    .filter(|&v| v >= 0.0 && (v as usize) < self.table.len())
                    .map(|v| v as usize)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("decide_comps {decide} selects no valid components"))
    }

    /// tedana `change_comptable_classifications` (both branches), then a status column.
    fn classify(
        &mut self,
        decision: &[(usize, bool)],
        if_true: &str,
        if_false: &str,
        tag_true: Option<&str>,
        tag_false: Option<&str>,
    ) -> (usize, usize) {
        for (state, class, tag) in [(true, if_true, tag_true), (false, if_false, tag_false)] {
            for &(c, _) in decision.iter().filter(|(_, b)| *b == state) {
                if class != "nochange" {
                    self.table.classification[c] = class.into();
                }
                for t in tag.into_iter().flat_map(|t| t.split(',')).map(str::trim) {
                    if !self.table.tags[c].iter().any(|x| x == t) {
                        self.table.tags[c].push(t.into());
                    }
                }
            }
        }
        self.status.push((
            format!("Node {}", self.node),
            self.table.classification.clone(),
        ));
        let n_true = decision.iter().filter(|(_, b)| *b).count();
        (n_true, decision.len() - n_true)
    }

    /// `classify` with the node's if_true/if_false classes and tags.
    fn classify_if(
        &mut self,
        decision: &[(usize, bool)],
        ps: impl Fn(&str) -> Option<String>,
    ) -> Result<(usize, usize), String> {
        let (it, iff) = (
            ps("if_true").ok_or("if_true missing")?,
            ps("if_false").ok_or("if_false missing")?,
        );
        Ok(self.classify(
            decision,
            &it,
            &iff,
            ps("tag_if_true").as_deref(),
            ps("tag_if_false").as_deref(),
        ))
    }

    fn run(&mut self, node: &Json) -> Result<Json, String> {
        let fname = node
            .get("functionname")
            .and_then(Json::as_str)
            .ok_or("node without functionname")?;
        // check_null: null parameters come from the cross-component metrics.
        let mut params: Vec<(String, Json)> = Vec::new();
        for group in ["parameters", "kwargs"] {
            if let Some(Json::Obj(kv)) = node.get(group) {
                for (k, v) in kv {
                    let v = if *v == Json::Null {
                        self.cross_value(k).cloned().ok_or(format!(
                            "Parameter {k} is required in node {fname}, but not defined"
                        ))?
                    } else {
                        v.clone()
                    };
                    params.push((k.clone(), v));
                }
            }
        }
        let p = |k: &str| params.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        let ps = |k: &str| p(k).and_then(|v| v.as_str().map(String::from));
        let pf = |k: &str, d: f64| p(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let custom_label = ps("custom_node_label").filter(|s| !s.is_empty());
        let node_label = |default: String| {
            let text = custom_label.clone().unwrap_or(default);
            ("node_label".to_string(), Json::Str(text))
        };
        let decide = p("decide_comps").unwrap_or(Json::Null);
        let idx = int(self.node);
        let used = |key: &str, m: &[&str]| {
            let names: BTreeSet<String> = m.iter().map(|s| s.to_string()).collect();
            (key.to_string(), Json::strings(&names))
        };
        let mut out: Vec<(String, Json)> = vec![("decision_node_idx".into(), idx)];
        let set_counts = |out: &mut Vec<(String, Json)>, t: usize, f: usize| {
            out.push(("n_true".into(), int(t)));
            out.push(("n_false".into(), int(f)));
        };

        match fname {
            "manual_classify" => {
                let new = ps("new_classification").ok_or("new_classification missing")?;
                out.push(node_label(format!("Set {} to {new}", py_str(&decide))));
                out.push(used("used_metrics", &[]));
                let comps = self.comps(&decide)?;
                let (t, f) = if comps.is_empty() {
                    (0, 0)
                } else {
                    let decision: Vec<(usize, bool)> = comps.iter().map(|&c| (c, true)).collect();
                    self.classify(&decision, &new, "nochange", ps("tag").as_deref(), None)
                };
                set_counts(&mut out, t, f);
                if p("clear_classification_tags").is_some_and(|v| v == Json::Bool(true)) {
                    self.table.tags.iter_mut().for_each(Vec::clear);
                }
            }
            "dec_left_op_right" => {
                let comps = self.comps(&decide)?;
                let (mut used_m, mut used_x) = (BTreeSet::new(), BTreeSet::new());
                let mut labels = Vec::new();
                let mut decision = vec![true; comps.len()];
                for suffix in ["", "2", "3"] {
                    let (op, left, right) = (
                        p(&format!("op{suffix}")),
                        p(&format!("left{suffix}")),
                        p(&format!("right{suffix}")),
                    );
                    if op.is_none() && left.is_none() && right.is_none() {
                        continue;
                    }
                    let (Some(op), Some(left), Some(right)) = (op, left, right) else {
                        return Err("left_op_right needs op, left and right together".into());
                    };
                    let op = op.as_str().ok_or("op must be a string")?.to_string();
                    let mut side = |val: &Json,
                                    scale_key: String|
                     -> Result<(Vec<f64>, String), String> {
                        let scale_json = p(&scale_key).unwrap_or(Json::Int(1));
                        let scale = match &scale_json {
                            Json::Str(name) => {
                                used_x.insert(name.clone());
                                self.cross_f64(name)?
                            }
                            v => v.as_f64().ok_or(format!("{scale_key} must be a number"))?,
                        };
                        let scale_text = match &scale_json {
                            Json::Str(name) => py_operand(self.cross_value(name).unwrap(), scale),
                            v => py_operand(v, scale),
                        };
                        let (vals, text) = match val {
                            Json::Str(name) if self.table.col(name).is_some() => {
                                used_m.insert(name.clone());
                                let col = self.table.col(name).unwrap();
                                (comps.iter().map(|&c| col[c]).collect(), name.clone())
                            }
                            // tedana only logs an unknown name when no components remain.
                            Json::Str(name)
                                if comps.is_empty() && self.cross_value(name).is_none() =>
                            {
                                (Vec::new(), name.clone())
                            }
                            Json::Str(name) => {
                                let v = self.cross_f64(name).map_err(|_| format!("{name} is neither a metric in selector.component_table_ nor selector.cross_component_metrics_"))?;
                                used_x.insert(name.clone());
                                (
                                    vec![v; comps.len()],
                                    py_operand(self.cross_value(name).unwrap(), v),
                                )
                            }
                            other => {
                                let v = other
                                    .as_f64()
                                    .ok_or("left/right must be a metric name or number")?;
                                (vec![v; comps.len()], py_operand(other, v))
                            }
                        };
                        let text = if scale == 1.0 {
                            text
                        } else {
                            format!("{scale_text}*{text}")
                        };
                        Ok((vals.iter().map(|v| scale * v).collect(), text))
                    };
                    let (l, lt) = side(&left, format!("left{suffix}_scale"))?;
                    let (r, rt) = side(&right, format!("right{suffix}_scale"))?;
                    labels.push(format!("{lt}{op}{rt}"));
                    for k in 0..comps.len() {
                        decision[k] = decision[k] && compare(l[k], &op, r[k])?;
                    }
                }
                let joined = if labels.len() == 1 {
                    Json::Str(labels[0].clone())
                } else {
                    Json::Arr(vec![Json::Str(labels.join(" & "))])
                };
                out.push(("used_metrics".into(), Json::strings(&used_m)));
                out.push((
                    "used_cross_component_metrics".into(),
                    Json::strings(&used_x),
                ));
                out.push((
                    "node_label".into(),
                    custom_label.clone().map_or(joined, Json::Str),
                ));
                let (t, f) = if comps.is_empty() {
                    (0, 0)
                } else {
                    let d: Vec<(usize, bool)> = comps.iter().copied().zip(decision).collect();
                    self.classify_if(&d, ps)?
                };
                set_counts(&mut out, t, f);
            }
            "dec_variance_lessthan_thresholds" => {
                let metric = ps("var_metric").unwrap_or("variance explained".into());
                let (single, all) = (
                    pf("single_comp_threshold", 0.1),
                    pf("all_comp_threshold", 1.0),
                );
                // f-string of the parameter: ints as given, floats (and the 0.1 / 1.0 defaults) as repr.
                let label_num = |k: &str, v: f64| match p(k) {
                    Some(Json::Int(i)) => i.to_string(),
                    _ => crate::json::py_float(v),
                };
                out.push(used("used_metrics", &[&metric]));
                out.push(node_label(format!(
                    "{metric}<{}. All variance<{}",
                    label_num("single_comp_threshold", single),
                    label_num("all_comp_threshold", all)
                )));
                let comps = self.comps(&decide)?;
                let (t, f) = if comps.is_empty() {
                    (0, 0)
                } else {
                    let var = self.values(&metric, &comps)?;
                    let mut d: Vec<bool> = var.iter().map(|&v| v < single).collect();
                    let total = |d: &[bool]| {
                        sum(&var
                            .iter()
                            .zip(d)
                            .filter(|(_, b)| **b)
                            .map(|(v, _)| *v)
                            .collect::<Vec<_>>())
                    };
                    while total(&d) > all {
                        let max = var
                            .iter()
                            .zip(&d)
                            .filter(|(_, b)| **b)
                            .map(|(v, _)| *v)
                            .fold(f64::NEG_INFINITY, f64::max);
                        var.iter()
                            .zip(d.iter_mut())
                            .filter(|(v, _)| **v == max)
                            .for_each(|(_, b)| *b = false);
                    }
                    let d: Vec<(usize, bool)> = comps.iter().copied().zip(d).collect();
                    self.classify_if(&d, ps)?
                };
                set_counts(&mut out, t, f);
            }
            "calc_median" => {
                let (metric, label) = (
                    ps("metric_name").ok_or("metric_name missing")?,
                    ps("median_label").ok_or("median_label missing")?,
                );
                let name = format!("median_{label}");
                out.push(node_label(format!("Median({name})")));
                out.push(used("used_metrics", &[&metric]));
                out.push(used("calc_cross_comp_metrics", &[&name]));
                let comps = self.comps(&decide)?;
                let value = if comps.is_empty() {
                    Json::Null
                } else {
                    let v = num(median(&self.values(&metric, &comps)?));
                    self.set_cross(&name, v.clone());
                    v
                };
                out.push((name, value));
            }
            "calc_kappa_elbow" => {
                let n_ind = self.elbow_outputs(&mut out, node_label("Calc Kappa Elbow".into()));
                out.push(used("used_metrics", &["kappa"]));
                out.push(used(
                    "calc_cross_comp_metrics",
                    &[
                        "kappa_elbow_kundu",
                        "kappa_allcomps_elbow",
                        "kappa_nonsig_elbow",
                        "varex_upper_p",
                    ],
                ));
                let comps = self.comps(&decide)?;
                let values = if comps.is_empty() {
                    None
                } else {
                    let kappas = self.values("kappa", &comps)?;
                    let f01 = f_ppf_dfn1(1.0 - 0.01, n_ind - 1.0);
                    let nonsig: Vec<f64> = kappas.iter().copied().filter(|&k| k < f01).collect();
                    let all_elbow = getelbow(&kappas);
                    let (elbow, nonsig_elbow) = if nonsig.len() >= 6 {
                        let e = getelbow(&nonsig);
                        (e.min(all_elbow), num(e))
                    } else {
                        (all_elbow, Json::Null)
                    };
                    // tedana reads positions within comps as table labels.
                    let varex = self
                        .table
                        .col("variance explained")
                        .ok_or("variance explained missing")?;
                    let high: Vec<f64> = (0..kappas.len())
                        .filter(|&i| kappas[i] > all_elbow)
                        .map(|i| varex[i])
                        .collect();
                    Some((elbow, all_elbow, nonsig_elbow, median(&high)))
                };
                let computed = values.is_some();
                let (a, b, c, d) = match values {
                    Some((e, ae, ne, vp)) => (num(e), num(ae), ne, num(vp)),
                    None => (Json::Null, Json::Null, Json::Null, Json::Null),
                };
                for (k, v) in [
                    ("kappa_elbow_kundu", a),
                    ("kappa_allcomps_elbow", b),
                    ("kappa_nonsig_elbow", c),
                    ("varex_upper_p", d),
                ] {
                    if computed {
                        self.publish(&mut out, k, v);
                    } else {
                        out.push((k.into(), v));
                    }
                }
            }
            "calc_rho_elbow" => {
                let kind = ps("rho_elbow_type").unwrap_or("kundu".into());
                let name = match kind.as_str() {
                    "kundu" => "rho_elbow_kundu",
                    "liberal" => "rho_elbow_liberal",
                    _ => {
                        return Err(format!(
                            "rho_elbow_type must be 'kundu' or 'liberal' It is {kind}"
                        ));
                    }
                };
                let n_ind = self.elbow_outputs(&mut out, node_label("Calc Rho Elbow".into()));
                out.push(used(
                    "calc_cross_comp_metrics",
                    &[
                        name,
                        "rho_allcomps_elbow",
                        "rho_unclassified_elbow",
                        "elbow_f05",
                    ],
                ));
                out.push(used(
                    "used_metrics",
                    &["kappa", "rho", "variance explained"],
                ));
                let comps = self.comps(&decide)?;
                let subset = self
                    .comps(&p("subset_decide_comps").unwrap_or(Json::Str("unclassified".into())))?;
                if comps.is_empty() {
                    for k in [
                        name,
                        "rho_allcomps_elbow",
                        "rho_unclassified_elbow",
                        "elbow_f05",
                    ] {
                        out.push((k.into(), Json::Null));
                    }
                    return Ok(Json::Obj(out));
                }
                let f05 = f_ppf_dfn1(1.0 - 0.05, n_ind - 1.0);
                let all_elbow = getelbow(&self.values("rho", &comps)?);
                let (elbow, uncl) = if subset.is_empty() {
                    (all_elbow, Json::Null)
                } else {
                    let u = getelbow(&self.values("rho", &subset)?);
                    let e = if kind == "kundu" {
                        (-0.0 + all_elbow + u + f05) / 3.0
                    } else {
                        all_elbow.max(u)
                    };
                    (e, num(u))
                };
                for (k, v) in [
                    (name, num(elbow)),
                    ("rho_allcomps_elbow", num(all_elbow)),
                    ("rho_unclassified_elbow", uncl),
                    ("elbow_f05", num(f05)),
                ] {
                    self.publish(&mut out, k, v);
                }
            }
            "dec_classification_doesnt_exist" => {
                let new = ps("new_classification").ok_or("new_classification missing")?;
                let exists = p("class_comp_exists").ok_or("class_comp_exists missing")?;
                let at_least = pf("at_least_num_exist", 1.0) as usize;
                let default = if at_least == 1 {
                    format!(
                        "Change {} to {new} if {} doesn't exist",
                        py_str(&decide),
                        py_str(&exists)
                    )
                } else {
                    format!(
                        "Change {} to {new} if less than {at_least} components with {} exist",
                        py_str(&decide),
                        py_str(&exists)
                    )
                };
                out.push(used("used_metrics", &[]));
                out.push(used("used_cross_comp_metrics", &[]));
                out.push(node_label(default));
                let comps = self.comps(&decide)?;
                let (t, f) = if comps.is_empty() || self.comps(&exists)?.len() >= at_least {
                    (0, comps.len())
                } else {
                    let d: Vec<(usize, bool)> = comps.iter().map(|&c| (c, true)).collect();
                    self.classify(&d, &new, "nochange", ps("tag").as_deref(), None)
                };
                set_counts(&mut out, t, f);
            }
            "dec_reclassify_high_var_comps" => {
                let new = ps("new_classification").ok_or("new_classification missing")?;
                out.push(used("used_metrics", &["variance explained"]));
                out.push(used("used_cross_comp_metrics", &["varex_upper_p"]));
                out.push(node_label(format!("Change {} to {new} for the top couple of components with the highest jumps in variance", py_str(&decide))));
                let comps = self.comps(&decide)?;
                let (t, f) = if comps.is_empty() {
                    (0, 0)
                } else {
                    let upper = self.cross_f64("varex_upper_p")?;
                    let varex = self
                        .table
                        .col("variance explained")
                        .ok_or("variance explained missing")?
                        .to_vec();
                    let mut keep = comps.clone();
                    for _ in 0..3 {
                        keep.sort_by(|&a, &b| varex[b].total_cmp(&varex[a]));
                        keep = (0..keep.len())
                            .filter(|&i| {
                                let diff = if i + 1 < keep.len() {
                                    varex[keep[i]] - varex[keep[i + 1]]
                                } else {
                                    0.0
                                };
                                diff < upper
                            })
                            .map(|i| keep[i])
                            .collect();
                    }
                    let d: Vec<(usize, bool)> =
                        comps.iter().map(|&c| (c, !keep.contains(&c))).collect();
                    self.classify(&d, &new, "nochange", ps("tag").as_deref(), None)
                };
                set_counts(&mut out, t, f);
            }
            "calc_varex_thresh" => {
                let label = ps("thresh_label").unwrap_or_default().to_lowercase();
                let (vname, pname) = if label.is_empty() {
                    ("varex_thresh".into(), "perc".into())
                } else {
                    (format!("varex_{label}_thresh"), format!("{label}_perc"))
                };
                let perc = p("percentile_thresh").ok_or("percentile_thresh missing")?;
                let per = perc
                    .as_f64()
                    .filter(|&v| v > 0.0 && v < 100.0)
                    .ok_or(format!(
                        "percentile_thresh must be a number between 0 & 100. It is: {perc}"
                    ))?;
                let highest = p("num_highest_var_comps").unwrap_or(Json::Null);
                out.push(node_label(format!(
                    "Calc {vname}, {}th percentile threshold",
                    py_str(&perc)
                )));
                out.push(("num_highest_var_comps".into(), highest.clone()));
                out.push(used("used_metrics", &["variance explained"]));
                out.push(used("calc_cross_comp_metrics", &[&vname, &pname]));
                out.push((pname, perc.clone()));
                let comps = self.comps(&decide)?;
                let value = if comps.is_empty() {
                    Json::Null
                } else {
                    let mut vals = self.values("variance explained", &comps)?;
                    if highest != Json::Null {
                        let n = match &highest {
                            Json::Str(name) => self.cross_f64(name)?,
                            other => other
                                .as_f64()
                                .ok_or("num_highest_var_comps must be an integer")?,
                        } as usize;
                        vals.sort_by(|a, b| b.total_cmp(a));
                        vals.truncate(n.min(vals.len()));
                    }
                    let v = num(score_at_percentile(&vals, per));
                    self.set_cross(&vname, v.clone());
                    v
                };
                out.push((vname, value));
            }
            "calc_extend_factor" => {
                let n_vols = self.cross_f64("n_vols")?;
                let factor = match p("extend_factor").and_then(|v| v.as_f64()) {
                    Some(f) if f != 0.0 => f,
                    _ if n_vols < 90.0 => 3.0,
                    _ if n_vols < 110.0 => 2.0 + (n_vols - 90.0) / 20.0,
                    _ => 2.0,
                };
                out.push(used("used_metrics", &[]));
                out.push(node_label("Calc extend_factor".into()));
                out.push(used("calc_cross_comp_metrics", &["extend_factor"]));
                self.publish(&mut out, "extend_factor", num(factor));
            }
            "calc_max_good_meanmetricrank" => {
                let name = match ps("metric_suffix").filter(|s| !s.is_empty()) {
                    Some(sfx) => format!("max_good_meanmetricrank_{sfx}"),
                    None => "max_good_meanmetricrank".into(),
                };
                out.push(node_label(format!("Calc {name}")));
                out.push(used("used_metrics", &[]));
                out.push(used("calc_cross_comp_metrics", &[&name]));
                let comps = self.comps(&decide)?;
                let value = if comps.is_empty() {
                    Json::Null
                } else {
                    let v = num(self.cross_f64("extend_factor")? * comps.len() as f64);
                    self.set_cross(&name, v.clone());
                    v
                };
                out.push((name, value));
            }
            "calc_varex_kappa_ratio" => {
                out.push(node_label("Calc varex kappa ratio".into()));
                out.push(used("used_metrics", &["kappa", "variance explained"]));
                out.push(used("calc_cross_comp_metrics", &["kappa_rate"]));
                out.push(used(
                    "added_component_table_metrics",
                    &["varex kappa ratio"],
                ));
                if self.table.col("varex kappa ratio").is_some() {
                    return Err(
                        "'varex kappa ratio' is already a column in the component_table".into(),
                    );
                }
                let comps = self.comps(&decide)?;
                let value = if comps.is_empty() {
                    Json::Null
                } else {
                    let (kappa, varex) = (
                        self.table.col("kappa").unwrap().to_vec(),
                        self.table.col("variance explained").unwrap().to_vec(),
                    );
                    let range = |v: &[f64]| {
                        let s: Vec<f64> = comps
                            .iter()
                            .map(|&c| v[c])
                            .filter(|x| !x.is_nan())
                            .collect();
                        s.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                            - s.iter().fold(f64::INFINITY, |a, &b| a.min(b))
                    };
                    let rate = range(&kappa) / range(&varex);
                    self.table.set_col(
                        "varex kappa ratio",
                        (0..kappa.len())
                            .map(|c| rate * varex[c] / kappa[c])
                            .collect(),
                    );
                    self.set_cross("kappa_rate", num(rate));
                    num(rate)
                };
                out.push(("kappa_rate".into(), value));
            }
            "calc_revised_meanmetricrank_guesses" => {
                let restrict = p("restrict_factor").unwrap_or(num(2.0));
                let rf = restrict.as_f64().ok_or(format!(
                    "restrict_factor needs to be a number. It is: {restrict}"
                ))?;
                let col_name = format!("d_table_score_node{}", self.node);
                if self.table.col(&col_name).is_some() {
                    return Err(format!(
                        "{col_name} is already a column in the component_table"
                    ));
                }
                out.push(node_label(
                    "Calc revised d_table_score & num accepted component guesses".into(),
                ));
                out.push(used(
                    "used_metrics",
                    &[
                        "kappa",
                        "dice_FT2",
                        "signal-noise_t",
                        "countnoise",
                        "countsigFT2",
                        "rho",
                    ],
                ));
                out.push(used(
                    "used_cross_component_metrics",
                    &["kappa_elbow_kundu", "rho_elbow_kundu"],
                ));
                out.push(used(
                    "calc_cross_comp_metrics",
                    &["num_acc_guess", "conservative_guess", "restrict_factor"],
                ));
                out.push(used("added_component_table_metrics", &[&col_name]));
                let comps = self.comps(&decide)?;
                if comps.is_empty() {
                    for k in ["num_acc_guess", "conservative_guess", "restrict_factor"] {
                        out.push((k.into(), Json::Null));
                    }
                } else {
                    let (ke, re) = (
                        self.cross_f64("kappa_elbow_kundu")?,
                        self.cross_f64("rho_elbow_kundu")?,
                    );
                    let pick = |name: &str| self.values(name, &comps);
                    let (kappa, rho) = (pick("kappa")?, pick("rho")?);
                    let both = (0..comps.len())
                        .filter(|&i| kappa[i] > ke && rho[i] < re)
                        .count();
                    let high = kappa.iter().filter(|&&k| k > ke).count();
                    let guess = ((-0.0 + both as f64 + high as f64) / 2.0) as i64 as f64;
                    let scores = decision_table_score(
                        &[&pick("countnoise")?],
                        &[
                            &kappa,
                            &pick("dice_FT2")?,
                            &pick("signal-noise_t")?,
                            &pick("countsigFT2")?,
                        ],
                    );
                    let mut column = vec![f64::NAN; self.table.len()];
                    for (k, &c) in comps.iter().enumerate() {
                        column[c] = scores[k];
                    }
                    self.table.set_col(&col_name, column);
                    for (k, v) in [
                        ("restrict_factor", restrict),
                        ("num_acc_guess", int(guess as usize)),
                        ("conservative_guess", num(guess / rf)),
                    ] {
                        self.publish(&mut out, k, v);
                    }
                }
            }
            other => return Err(format!("unsupported node function {other}")),
        }
        Ok(Json::Obj(out))
    }
}

/// Python `str()` of a decide_comps-style value, for node labels.
fn py_str(v: &Json) -> String {
    match v {
        Json::Str(s) => s.clone(),
        Json::Num(n) if n.fract() == 0.0 => format!("{}", *n as i64),
        Json::Arr(a) => format!(
            "[{}]",
            a.iter()
                .map(|x| if let Json::Str(s) = x {
                    format!("'{s}'")
                } else {
                    py_str(x)
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}

/// Python `str(round(v, 2))` of a float, for node labels (Rust's `{:.2}` rounds the exact value
/// half to even, as Python's round does).
fn py_round2(v: f64) -> String {
    crate::json::py_float(format!("{v:.2}").parse().unwrap())
}

/// A label operand as tedana prints it: ints unchanged, floats rounded to 2 places.
fn py_operand(v: &Json, value: f64) -> String {
    if let Json::Int(i) = v {
        i.to_string()
    } else {
        py_round2(value)
    }
}
