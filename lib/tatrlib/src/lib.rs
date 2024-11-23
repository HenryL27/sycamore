use std::{
    collections::{HashMap, HashSet},
    iter::FusedIterator,
    ops::Deref,
};

use pyo3::{
    exceptions::{PyKeyError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyFloat, PyList, PyString},
};

#[derive(Clone, Debug, PartialEq)]
enum Label {
    Table,
    Row,
    Column,
    ColumnHeader,
    ProjectedRowHeader,
    SpanningCell,
}

#[derive(Debug, FromPyObject)]
struct BBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[derive(Debug, FromPyObject)]
struct TableObject {
    #[pyo3(from_py_with = "convert_label")]
    pub label: Label,
    pub score: f64,
    pub bbox: BBox,
}

fn convert_label(pyvalue: &Bound<PyAny>) -> PyResult<Label> {
    Label::try_from(pyvalue)
}

#[derive(Debug)]
struct TableObjects {
    pub inner: Vec<TableObject>,
}

#[derive(Debug, FromPyObject)]
struct TableToken {
    pub bbox: BBox,
    pub span_num: i32,
    pub line_num: i32,
    pub block_num: i32,
    pub text: String,
}

struct ClassThresholds {
    table: f32,
    row: f32,
    column: f32,
    prh: f32,
    col_header: f32,
    spanning_cell: f32,
    no_object: f32,
}

#[pyfunction]
fn objects_to_table(
    objects: Vec<TableObject>,
    tokens: Vec<TableToken>,
    #[pyo3(from_py_with = "convert_class_thresholds")] structure_class_thresholds: ClassThresholds,
    union_tokens: bool,
) {
}

#[pyfunction]
fn objects_to_structures(
    objects: Vec<TableObject>,
    tokens: Vec<TableToken>,
    #[pyo3(from_py_with = "convert_class_thresholds")] class_thresholds: ClassThresholds,
) -> Option<()> {
    let first_table = objects.iter().find(|obj| obj.label == Label::Table)?;
    let objs_in_table: Vec<&TableObject> = objects
        .iter()
        .filter(|&obj| iob(&obj.bbox, &first_table.bbox) >= 0.5)
        .collect();
    let tokens_in_table: Vec<&TableToken> = tokens
        .iter()
        .filter(|&obj| iob(&obj.bbox, &first_table.bbox) >= 0.5)
        .collect();
    let mut columns: Vec<&TableObject> = objs_in_table
        .iter()
        .filter(|ob| ob.label == Label::Column)
        .map(|&ob| ob)
        .collect();
    let mut rows: Vec<&TableObject> = objs_in_table
        .iter()
        .filter(|ob| ob.label == Label::Row)
        .map(|&ob| ob)
        .collect();
    let column_headers: Vec<&TableObject> = objs_in_table
        .iter()
        .filter(|ob| ob.label == Label::ColumnHeader)
        .map(|&ob| ob)
        .collect();
    let spanning_cells: Vec<&TableObject> = objs_in_table
        .iter()
        .filter(|ob| ob.label == Label::SpanningCell)
        .map(|&ob| ob)
        .collect();
    let prhs: Vec<&TableObject> = objs_in_table
        .iter()
        .filter(|ob| ob.label == Label::ProjectedRowHeader)
        .map(|&ob| ob)
        .collect();
    let mut col_header_rows = Vec::new();
    for row in &rows {
        for col_header in &column_headers {
            if iob(&row.bbox, &col_header.bbox) >= 0.5 {
                col_header_rows.push(row);
            }
        }
    }
    // Refine table structures
    refine_rows(&mut rows, &tokens_in_table, class_thresholds.row);
    refine_columns(&mut columns, &tokens_in_table, class_thresholds.column);

    // Shrink table bbox to just the total height of the rows
    // and the total width of the columns
    let row_rect = if rows.len() > 0 {
        let x1 = rows.iter().map(|row| row.bbox.x1).min_by(f32::total_cmp)?;
        let x2 = rows.iter().map(|row| row.bbox.x2).max_by(f32::total_cmp)?;
        let y1 = rows.iter().map(|row| row.bbox.y1).min_by(f32::total_cmp)?;
        let y2 = rows.iter().map(|row| row.bbox.y2).min_by(f32::total_cmp)?;
        BBox { x1, x2, y1, y2 }
    } else {
        BBox {
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    };
    // Shorter lines -> autoformatter doesn't split each line into several
    let cols = columns;
    let col_rect = if cols.len() > 0 {
        let x1 = cols.iter().map(|row| row.bbox.x1).min_by(f32::total_cmp)?;
        let x2 = cols.iter().map(|row| row.bbox.x2).max_by(f32::total_cmp)?;
        let y1 = cols.iter().map(|row| row.bbox.y1).min_by(f32::total_cmp)?;
        let y2 = cols.iter().map(|row| row.bbox.y2).min_by(f32::total_cmp)?;
        BBox { x1, x2, y1, y2 }
    } else {
        BBox {
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    };
    let row_col_bbox = BBox {
        x1: col_rect.x1,
        y1: row_rect.y1,
        x2: col_rect.x2,
        y2: row_rect.y2,
    };
    first_table.bbox.x1 = row_col_bbox.x1;

    Some(())
}

fn refine_rows(mut rows: &mut Vec<&TableObject>, tokens: &Vec<&TableToken>, threshold: f32) {
    if tokens.len() > 0 {
        nms_by_containment(&mut rows, tokens, 0.5, true);
        remove_objects_without_content(tokens, &mut rows);
    } else {
        nms(&mut rows, NMSMatchCriteria::Object2Overlap, 0.5, true);
    }
    if rows.len() > 1 {
        rows.sort_by(|ob1, ob2| {
            (ob1.bbox.y1 + ob1.bbox.y2).total_cmp(&(ob2.bbox.y1 + ob2.bbox.y2))
        });
    }
}

fn refine_columns(mut cols: &mut Vec<&TableObject>, tokens: &Vec<&TableToken>, threshold: f32) {
    if tokens.len() > 0 {
        nms_by_containment(&mut cols, tokens, 0.5, false);
        remove_objects_without_content(tokens, &mut cols);
    } else {
        nms(&mut cols, NMSMatchCriteria::Object2Overlap, 0.5, true);
    }
    if cols.len() > 1 {
        cols.sort_by(|ob1, ob2| {
            (ob1.bbox.x1 + ob1.bbox.x2).total_cmp(&(ob2.bbox.x1 + ob2.bbox.x2))
        });
    }
}

enum NMSMatchCriteria {
    Object2Overlap,
    Object1Overlap,
    IOU,
}

fn nms(
    objects: &mut Vec<&TableObject>,
    match_criteria: NMSMatchCriteria,
    match_threshold: f32,
    keep_higher: bool,
) {
    if objects.len() == 0 {
        return;
    }

    objects.sort_by(|ob1, ob2| match keep_higher {
        true => (ob1.score * -1.0).total_cmp(&(ob2.score * -1.0)),
        false => ob1.score.total_cmp(&ob2.score),
    });

    let num_objects = objects.len();
    let mut suppression = Vec::with_capacity(num_objects);
    for _ in 0..num_objects {
        suppression.push(false);
    }

    for object2_num in 1..num_objects {
        let ob2_rect = &objects[object2_num].bbox;
        let ob2_area = ob2_rect.area();
        for object1_num in 0..object2_num {
            if !suppression[object1_num] {
                let ob1_rect = &objects[object1_num].bbox;
                let ob1_area = ob1_rect.area();
                let intersect_area = ob1_rect.intersect(ob2_rect).area();
                let metric = match match_criteria {
                    NMSMatchCriteria::Object1Overlap => divide_or_zero(intersect_area, ob1_area),
                    NMSMatchCriteria::Object2Overlap => divide_or_zero(intersect_area, ob2_area),
                    NMSMatchCriteria::IOU => {
                        divide_or_zero(intersect_area, ob1_area + ob2_area - intersect_area)
                    }
                };
                if metric >= match_threshold {
                    suppression[object2_num] = true;
                    break;
                }
            }
        }
    }

    for i in objects.len() - 1..0 {
        if suppression[i] {
            objects.remove(i);
        }
    }
}

fn divide_or_zero(a: f32, b: f32) -> f32 {
    match b {
        0.0 => 0.0,
        b => a / b,
    }
}

fn remove_objects_without_content(page_spans: &Vec<&TableToken>, objects: &mut Vec<&TableObject>) {
    for (i, obj) in objects.clone().iter().enumerate().rev() {
        let (ob_text, _) = extract_text_inside_bbox(page_spans, &obj.bbox);
        if ob_text.trim().len() == 0 {
            objects.remove(i);
        }
    }
}

fn extract_text_inside_bbox<'a>(
    spans: &Vec<&'a TableToken>,
    bbox: &BBox,
) -> (String, Vec<&'a TableToken>) {
    let bb_spans = get_bbox_span_subset(spans, bbox, 0.5);
    let bb_text = extract_text_from_spans(&bb_spans, true, true);
    (bb_text, bb_spans)
}

fn get_bbox_span_subset<'a>(
    spans: &Vec<&'a TableToken>,
    bbox: &BBox,
    threshold: f32,
) -> Vec<&'a TableToken> {
    let mut span_subset = Vec::new();
    for span in spans {
        if span.bbox.overlaps(bbox, threshold) {
            span_subset.push(span.clone());
        }
    }
    return span_subset;
}

fn extract_text_from_spans(
    spans: &Vec<&TableToken>,
    join_with_space: bool,
    remove_integer_subscripts: bool,
) -> String {
    let join_char = if join_with_space { " " } else { "" };
    let mut spans_copy = Vec::with_capacity(spans.len());
    spans_copy.copy_from_slice(spans.as_slice());

    // The original implementation looks at flags on the tokens to potentially
    // throw out integer subscripts (when remove_integer_subscripts is true).
    // We ain't doin that.

    spans_copy.sort_by_key(|sp| sp.span_num);
    spans_copy.sort_by_key(|sp| sp.line_num);
    spans_copy.sort_by_key(|sp| sp.block_num);

    // Force the span at the end of every line within a block to have ecatly one space
    // unless the line ends with a space or ends with a non-space followed by a hyphen
    let mut line_texts = Vec::new();
    let mut line_span_texts = [&spans_copy[0].text].to_vec();
    for (&span1, &span2) in spans_copy[..spans_copy.len() - 1]
        .iter()
        .zip(&spans_copy[1..])
    {
        if span1.block_num != span2.block_num || span1.line_num != span2.line_num {
            let line_text = line_span_texts
                .iter()
                .map(|&s| s.as_str())
                .collect::<Vec<&str>>()
                .join(join_char)
                .trim()
                .to_string();
            if line_text.len() > 0
                && !line_text.ends_with(" ")
                && !(line_text.len() > 1
                    && line_text.ends_with("-")
                    && !line_text[line_text.len() - 2..].starts_with(" "))
            {
                if !join_with_space {
                    line_texts.push(line_text.to_string() + " ");
                    line_span_texts = [&span2.text].to_vec();
                } else {
                    line_span_texts.push(&span2.text);
                }
            }
        }
    }
    let line_text = line_span_texts
        .iter()
        .map(|&s| s.as_str())
        .collect::<Vec<&str>>()
        .join(join_char)
        .to_string();
    line_texts.push(line_text);

    return line_texts.join(join_char).trim().to_string();
}

fn nms_by_containment(
    container_objects: &mut Vec<&TableObject>,
    package_objects: &Vec<&TableToken>,
    overlap_threshold: f32,
    _early_exit_vertical: bool,
) {
    container_objects.sort_by(|toa, tob| (toa.score * -1.0).total_cmp(&(tob.score * -1.0)));
    let num_objects = container_objects.len();
    let mut suppression = Vec::with_capacity(num_objects);
    for _ in 0..num_objects {
        suppression.push(false);
    }
    let (packages_by_container, _, _) = slot_into_containers(
        container_objects,
        package_objects,
        overlap_threshold,
        true,
        false,
        _early_exit_vertical,
    );

    for object2_num in 1..num_objects {
        let object2_packages: HashSet<&usize> =
            HashSet::from_iter(packages_by_container[object2_num].iter());
        if object2_packages.len() == 0 {
            suppression[object2_num] = true;
        }
        for object1_num in 0..object2_num {
            if !suppression[object1_num] {
                let object1_packages =
                    HashSet::from_iter(packages_by_container[object1_num].iter());
                if !object2_packages.is_disjoint(&object1_packages) {
                    suppression[object2_num] = true;
                }
            }
        }
    }
    for i in container_objects.len() - 1..0 {
        if suppression[i] {
            container_objects.remove(i);
        }
    }
}

// fn nms_by_containment(
//     container_objects: &Vec<TableObject>,
//     package_objects: &Vec<TableToken>,
//     overlap_threshold: f23,
// ) -> Vec<TableObject> {
//     nms_by_containment(container_objects, package_objects, overlap_threshold, true)
// }

#[derive(Clone)]
struct MatchScore<'a> {
    pub container: &'a TableObject,
    pub container_num: usize,
    pub score: f32,
}

fn slot_into_containers(
    container_objects: &Vec<&TableObject>,
    package_objects: &Vec<&TableToken>,
    overlap_threshold: f32,
    unique_assignment: bool,
    forced_assignment: bool,
    _early_exit_vertical: bool,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<f32>) {
    let mut best_match_scores = Vec::new();
    let mut container_assignments = Vec::with_capacity(container_objects.len());
    let mut package_assignments = Vec::with_capacity(package_objects.len());
    for _ in 0..container_objects.len() {
        container_assignments.push(Vec::new());
    }
    for _ in 0..package_objects.len() {
        package_assignments.push(Vec::new());
    }
    if container_assignments.len() == 0 || package_objects.len() == 0 {
        return (
            container_assignments,
            package_assignments,
            best_match_scores,
        );
    }

    let sorted_co = match _early_exit_vertical {
        true => {
            let mut x = container_objects
                .iter()
                .enumerate()
                .map(|(i, &ob)| (i, ob))
                .collect::<Vec<(usize, &TableObject)>>();
            x.sort_by(|(_, toa), (_, tob)| toa.bbox.y1.total_cmp(&tob.bbox.y1));
            x
        }
        false => {
            let mut x = container_objects
                .iter()
                .enumerate()
                .map(|(i, &ob)| (i, ob))
                .collect::<Vec<(usize, &TableObject)>>();
            x.sort_by(|(_, toa), (_, tob)| toa.bbox.x1.total_cmp(&tob.bbox.x1));
            x
        }
    };

    for (pi, package) in package_objects.iter().enumerate() {
        let mut match_scores = Vec::new();
        let package_rect = &package.bbox;
        if package_rect.is_empty() {
            continue;
        }
        let package_area = package_rect.area();
        for (ci, container) in &sorted_co {
            if !_early_exit_vertical && container.bbox.x1 > container.bbox.x2 {
                if match_scores.len() == 0 {
                    match_scores.push(MatchScore {
                        container: &container,
                        container_num: *ci,
                        score: 0.0,
                    });
                }
                break;
            } else if _early_exit_vertical && container.bbox.y1 > container.bbox.y2 {
                if match_scores.len() == 0 {
                    match_scores.push(MatchScore {
                        container: &container,
                        container_num: *ci,
                        score: 0.0,
                    });
                }
                break;
            }
            let container_rect = &container.bbox;
            let intersect_area = container_rect.intersect(package_rect).area();
            let overlap_fraction = intersect_area / package_area;
            match_scores.push(MatchScore {
                container: &container,
                container_num: *ci,
                score: overlap_fraction,
            });
        }

        let sorted_match_scores = match unique_assignment {
            true => {
                let mut x = Vec::with_capacity(1);
                let mut best = match_scores[0].clone();
                for ms in match_scores {
                    if ms.score > best.score {
                        best = ms;
                    }
                }
                x.push(best);
                x
            }
            false => {
                match_scores.sort_by(|msa, msb| msa.score.total_cmp(&msb.score));
                match_scores
            }
        };

        let best_match_score = sorted_match_scores[0].clone();
        best_match_scores.push(best_match_score.score);
        if forced_assignment || best_match_score.score >= overlap_threshold {
            container_assignments[best_match_score.container_num].push(pi);
            package_assignments[pi].push(best_match_score.container_num);
        }
        if !unique_assignment {
            for match_score in &sorted_match_scores[1..] {
                if match_score.score >= overlap_threshold {
                    container_assignments[match_score.container_num].push(pi);
                    package_assignments[pi].push(match_score.container_num);
                } else {
                    break;
                }
            }
        }
    }

    (
        container_assignments,
        package_assignments,
        best_match_scores,
    )
}

impl BBox {
    fn empty() -> Self {
        BBox {
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn iob(self: &Self, other: &BBox) -> f32 {
        iob(self, other)
    }

    fn is_empty(self: &Self) -> bool {
        self.x1 >= self.x2 || self.y1 >= self.y2
    }

    fn area(self: &Self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn intersect(self: &Self, other: &Self) -> Self {
        let x1 = if self.x1 > other.x1 {
            other.x1
        } else {
            self.x1
        };
        let x2 = if self.x2 < other.x2 {
            other.x2
        } else {
            self.x2
        };
        let y1 = if self.y1 > other.y1 {
            other.y1
        } else {
            self.y2
        };
        let y2 = if self.y2 < other.y2 {
            other.y2
        } else {
            self.y2
        };

        if x1 >= x2 || y1 >= y2 {
            return BBox::empty();
        }
        return BBox { x1, y1, x2, y2 };
    }

    fn overlaps(self: &Self, other: &Self, threshold: f32) -> bool {
        let myarea = self.area();
        if myarea == 0.0 {
            return false;
        }
        return self.intersect(other).area() / myarea >= threshold;
    }
}

fn iob(bb1: &BBox, bb2: &BBox) -> f32 {
    let x1 = if bb1.x1 < bb2.x1 { bb1.x1 } else { bb2.x1 };
    let y1 = if bb1.y1 < bb2.y1 { bb1.y1 } else { bb2.y1 };
    let x2 = if bb1.x2 > bb2.x2 { bb1.x2 } else { bb2.x2 };
    let y2 = if bb1.y2 > bb2.y2 { bb1.y2 } else { bb2.y2 };
    if x1 > x2 || y1 > y2 || ((bb1.x2 - bb1.x1) * (bb1.y2 - bb1.y2)) == 0.0 {
        return 0.0;
    } else {
        return ((x2 - x1) * (y2 - y1)) / ((bb1.x2 - bb1.x1) * (bb1.y2 - bb1.y1));
    }
}

fn convert_class_thresholds(pyvalue: &Bound<PyAny>) -> PyResult<ClassThresholds> {
    let asdict = pyvalue.downcast::<PyDict>()?;
    Ok(ClassThresholds {
        table: asdict.get_item("table")?.map_or(Ok(0.5), |x| x.extract())?,
        row: asdict
            .get_item("table row")?
            .map_or(Ok(0.5), |x| x.extract())?,
        column: asdict
            .get_item("table column")?
            .map_or(Ok(0.5), |x| x.extract())?,
        prh: asdict
            .get_item("table projected row header")?
            .map_or(Ok(0.5), |x| x.extract())?,
        col_header: asdict
            .get_item("table column header")?
            .map_or(Ok(0.5), |x| x.extract())?,
        spanning_cell: asdict
            .get_item("table spanning cell")?
            .map_or(Ok(0.5), |x| x.extract())?,
        no_object: asdict
            .get_item("no object")?
            .map_or(Ok(10.0), |x| x.extract())?,
    })
}

impl TryFrom<&Bound<'_, PyList>> for TableObjects {
    type Error = PyErr;
    fn try_from(value: &Bound<'_, PyList>) -> Result<Self, Self::Error> {
        let l = value.len();
        let mut v = Vec::with_capacity(l);
        for i in 1..l {
            v[i] = TableObject::try_from(value.get_item(i))?;
        }
        return Ok(TableObjects { inner: v });
    }
}

impl TryFrom<PyResult<Bound<'_, PyAny>>> for TableObject {
    type Error = PyErr;

    fn try_from(value: PyResult<Bound<'_, PyAny>>) -> Result<Self, Self::Error> {
        match value {
            Err(e) => Err(e),
            Ok(object) => match object.downcast_into::<PyDict>() {
                Err(_) => Err(PyTypeError::new_err("Table objects should be dicts")),
                Ok(dict) => {
                    let label = match dict.get_item("label") {
                        Err(_) => Err(PyKeyError::new_err("\"label\" key not found")),
                        Ok(None) => Err(PyValueError::new_err("label was None")),
                        Ok(Some(x)) => Label::try_from(x),
                    }?;
                    let score = match dict.get_item("score") {
                        Err(_) => Err(PyKeyError::new_err("\"score\" key not found")),
                        Ok(None) => Err(PyValueError::new_err("score was None")),
                        Ok(Some(x)) => match x.downcast::<PyFloat>() {
                            Err(_) => Err(PyTypeError::new_err(
                                "Could not convert score to rust float",
                            )),
                            Ok(f) => Ok(f.value()),
                        },
                    }?;
                    let bbox = match dict.get_item("bbox") {
                        Err(_) => Err(PyKeyError::new_err("\"bbox\" key not found")),
                        Ok(None) => Err(PyValueError::new_err("bbox was None")),
                        Ok(Some(x)) => BBox::try_from(x),
                    }?;
                    return Ok(TableObject { label, score, bbox });
                }
            },
        }
    }
}

impl TryFrom<Bound<'_, PyAny>> for Label {
    type Error = PyErr;

    fn try_from(value: Bound<'_, PyAny>) -> Result<Self, Self::Error> {
        match value.downcast_into::<PyString>() {
            Err(_) => Err(PyTypeError::new_err("\"label\" key must be a string")),
            Ok(pystr) => match pystr.to_str() {
                Err(_) => Err(PyTypeError::new_err(
                    "Failed to convert python string to rust string",
                )),
                Ok("table") => Ok(Label::Table),
                Ok("table row") => Ok(Label::Row),
                Ok("table column") => Ok(Label::Column),
                Ok("table column header") => Ok(Label::ColumnHeader),
                Ok("table projected row header") => Ok(Label::ProjectedRowHeader),
                Ok("table spanning cell") => Ok(Label::SpanningCell),
                Ok(other) => Err(PyTypeError::new_err(format!("Invalid label: {other}"))),
            },
        }
    }
}

impl TryFrom<&Bound<'_, PyAny>> for Label {
    type Error = PyErr;

    fn try_from(value: &Bound<'_, PyAny>) -> Result<Self, Self::Error> {
        match value.extract::<&str>() {
            Err(_) => Err(PyTypeError::new_err("\"label\" key must be a string")),
            Ok("table") => Ok(Label::Table),
            Ok("table row") => Ok(Label::Row),
            Ok("table column") => Ok(Label::Column),
            Ok("table column header") => Ok(Label::ColumnHeader),
            Ok("table projected row header") => Ok(Label::ProjectedRowHeader),
            Ok("table spanning cell") => Ok(Label::SpanningCell),
            Ok(other) => Err(PyTypeError::new_err(format!("Invalid label: {other}"))),
        }
    }
}

impl TryFrom<Bound<'_, PyAny>> for BBox {
    type Error = PyErr;

    fn try_from(value: Bound<'_, PyAny>) -> Result<Self, Self::Error> {
        match value.downcast::<PyList>() {
            Err(_) => Err(PyTypeError::new_err("bbox must be a list")),
            Ok(pylist) => {
                if pylist.len() != 4 {
                    return Err(PyValueError::new_err("bbox must have length 4"));
                }
                let (x1, y1, x2, y2) = unsafe {
                    (
                        pylist.get_item_unchecked(0),
                        pylist.get_item_unchecked(1),
                        pylist.get_item_unchecked(2),
                        pylist.get_item_unchecked(3),
                    )
                };
                let x1 = x1.extract::<f32>();
                let y1 = y1.extract::<f32>();
                let x2 = x2.extract::<f32>();
                let y2 = y2.extract::<f32>();
                match (x1, y1, x2, y2) {
                    (Ok(x1v), Ok(y1v), Ok(x2v), Ok(y2v)) => Ok(BBox {
                        x1: x1v,
                        y1: y1v,
                        x2: x2v,
                        y2: y2v,
                    }),
                    (_, _, _, _) => Err(PyTypeError::new_err("bbox has non-float values")),
                }
            }
        }
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn tatrlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(objects_to_structures, m)?)?;
    Ok(())
}
