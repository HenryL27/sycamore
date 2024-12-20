use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
    iter::FusedIterator,
    ops::{Deref, DerefMut},
};

use pyo3::{
    exceptions::{PyKeyError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyFloat, PyList, PyString},
};

#[derive(Clone, Debug, PartialEq, Copy)]
enum Label {
    Table,
    Row,
    Column,
    ColumnHeader,
    ProjectedRowHeader,
    SpanningCell,
}

#[derive(Debug, FromPyObject, Clone, Copy)]
struct BBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[derive(Debug, FromPyObject, Clone, Copy)]
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

struct TableStructure {
    rows: Vec<RefCell<TableObject>>,
    cols: Vec<RefCell<TableObject>>,
    col_headers: Vec<RefCell<TableObject>>,
    spans: Vec<RefCell<TableObject>>,
    prhs: Vec<RefCell<TableObject>>,
}

struct ClassThresholds {
    table: f64,
    row: f64,
    column: f64,
    prh: f64,
    col_header: f64,
    spanning_cell: f64,
    no_object: f64,
}

#[pyfunction]
fn objects_to_table(
    objects: Vec<TableObject>,
    tokens: Vec<TableToken>,
    #[pyo3(from_py_with = "convert_class_thresholds")] structure_class_thresholds: ClassThresholds,
    union_tokens: bool,
) {
}

fn copy_objects(objects: &mut Vec<TableObject>) -> Vec<RefCell<TableObject>> {
    let mutable_objects = &mut Vec::with_capacity(objects.len());
    for o in objects {
        let copied = RefCell::new(TableObject {
            label: o.label.clone(),
            score: o.score.clone(),
            bbox: o.bbox.clone(),
        });
        mutable_objects.push(copied);
    }
    mutable_objects.to_owned()
}

fn copy_objects_mut(objects: &mut Vec<&mut TableObject>) -> Vec<RefCell<TableObject>> {
    let mut mutable_objects = Vec::with_capacity(objects.len());
    for o in objects {
        let copied = RefCell::new(TableObject {
            label: o.label.clone(),
            score: o.score.clone(),
            bbox: o.bbox.clone(),
        });
        mutable_objects.push(copied)
    }
    mutable_objects
}

#[pyfunction]
fn objects_to_structures<'a>(
    mut objects: Vec<TableObject>,
    tokens: Vec<TableToken>,
    #[pyo3(from_py_with = "convert_class_thresholds")] class_thresholds: ClassThresholds,
) -> Option<()> {
    let mutable_objects = copy_objects(&mut objects);
    let mut first_table: Option<RefCell<TableObject>> = None;
    for ob in &mutable_objects {
        if ob.borrow().label == Label::Table {
            first_table = Some(ob.clone());
            break;
        }
    }
    let first_table = first_table?;

    let objs_in_table: Vec<RefCell<TableObject>> = mutable_objects
        .iter()
        .filter(|obj| iob(&obj.borrow().bbox, &first_table.borrow().bbox) >= 0.5)
        .map(|obj| obj.clone())
        .collect();
    let tokens_in_table: Vec<&TableToken> = tokens
        .iter()
        .filter(|&obj| iob(&obj.bbox, &first_table.borrow().bbox) >= 0.5)
        .collect();
    let mut columns: Vec<RefCell<TableObject>> = objs_in_table
        .iter()
        .filter(|ob| ob.borrow().label == Label::Column)
        .map(|ob| ob.clone())
        .collect();
    let mut rows: Vec<RefCell<TableObject>> = objs_in_table
        .iter()
        .filter(|ob| ob.borrow().label == Label::Row)
        .map(|ob| ob.clone())
        .collect();
    let column_headers: Vec<RefCell<TableObject>> = objs_in_table
        .iter()
        .filter(|ob| ob.borrow().label == Label::ColumnHeader)
        .map(|ob| ob.clone())
        .collect();
    let spanning_cells: Vec<RefCell<TableObject>> = objs_in_table
        .iter()
        .filter(|ob| ob.borrow().label == Label::SpanningCell)
        .map(|ob| ob.clone())
        .collect();
    let prhs: Vec<RefCell<TableObject>> = objs_in_table
        .iter()
        .filter(|ob| ob.borrow().label == Label::ProjectedRowHeader)
        .map(|ob| ob.clone())
        .collect();
    let mut col_header_rows = Vec::new();
    for row in &rows {
        for col_header in &column_headers {
            if iob(&row.borrow().bbox, &col_header.borrow().bbox) >= 0.5 {
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
        let x1 = rows
            .iter()
            .map(|row| row.borrow().bbox.x1)
            .min_by(f32::total_cmp)?;
        let x2 = rows
            .iter()
            .map(|row| row.borrow().bbox.x2)
            .max_by(f32::total_cmp)?;
        let y1 = rows
            .iter()
            .map(|row| row.borrow().bbox.y1)
            .min_by(f32::total_cmp)?;
        let y2 = rows
            .iter()
            .map(|row| row.borrow().bbox.y2)
            .min_by(f32::total_cmp)?;
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
    let cols = &columns;
    let col_rect = if cols.len() > 0 {
        let x1 = cols
            .iter()
            .map(|row| row.borrow().bbox.x1)
            .min_by(f32::total_cmp)?;
        let x2 = cols
            .iter()
            .map(|row| row.borrow().bbox.x2)
            .max_by(f32::total_cmp)?;
        let y1 = cols
            .iter()
            .map(|row| row.borrow().bbox.y1)
            .min_by(f32::total_cmp)?;
        let y2 = cols
            .iter()
            .map(|row| row.borrow().bbox.y2)
            .min_by(f32::total_cmp)?;
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
    first_table.borrow_mut().bbox = row_col_bbox;

    align_columns(&mut columns, &row_col_bbox);
    align_rows(&mut rows, &row_col_bbox);

    if rows.is_empty() && row_rect.y1 < row_rect.y2 && col_rect.x1 < col_rect.x2 {
        rows.push(RefCell::new(TableObject {
            label: Label::Row,
            score: 0.0001,
            bbox: row_col_bbox,
        }));
    }
    if columns.is_empty() && row_rect.y1 < row_rect.y2 && col_rect.x1 < col_rect.x2 {
        columns.push(RefCell::new(TableObject {
            label: Label::Column,
            score: 0.0001,
            bbox: row_col_bbox,
        }));
    }
    let structure = &mut TableStructure {
        rows: rows,
        cols: columns,
        col_headers: column_headers,
        spans: spanning_cells,
        prhs: prhs,
    };

    if structure.cols.len() > 1 && structure.rows.len() > 0 {
        refine_table_structure(structure, &class_thresholds);
    }

    None
}

fn refine_table_structure(structure: &mut TableStructure, thresholds: &ClassThresholds) {
    let rows = &mut structure.rows;
    let cols = &mut structure.cols;

    // column headers
    let col_headers = &mut structure.col_headers;
    apply_threshold(col_headers, thresholds.col_header);
    nms(col_headers, NMSMatchCriteria::Object2Overlap, 0.05, true);
    let mut header_row_nums = align_headers(col_headers, rows);
    header_row_nums.sort();

    // spanning cells
    let spans = &mut structure.spans;
    let prhs = &mut structure.prhs;
    apply_threshold(spans, thresholds.spanning_cell);
    apply_threshold(prhs, thresholds.prh);
    spans.append(prhs);

    let supercells = &mut align_supercells(spans, rows, cols, header_row_nums);
    nms_supercells(supercells);
    header_supercell_tree(supercells);
}

fn header_supercell_tree(supercells: &mut Vec<RefCell<TableCell>>) {}

fn nms_supercells(supercells: &mut Vec<RefCell<TableCell>>) {
    supercells.sort_by(|a, b| {
        a.borrow()
            .inner
            .borrow()
            .score
            .total_cmp(&b.borrow().inner.borrow().score)
    });
    let num_supercells = supercells.len();
    let mut suppression = Vec::with_capacity(num_supercells);
    suppression.fill(false);
    for sc2_num in 1..num_supercells {
        let sc2 = &supercells[sc2_num];
        for sc1_num in 0..sc2_num {
            let sc1 = &supercells[sc1_num];
            remove_supercell_overlap(sc1, sc2);
        }
        if sc2.borrow().cols.len() * sc2.borrow().rows.len() < 2 {
            suppression[sc2_num] = true;
        }
    }
    for i in (0..num_supercells).rev() {
        if suppression[i] {
            supercells.remove(i);
        }
    }
}

fn remove_supercell_overlap(sc1: &RefCell<TableCell>, sc2: &RefCell<TableCell>) {
    let sc1_rns = &mut sc1.borrow_mut().rows;
    let sc1_cns = &mut sc1.borrow_mut().cols;
    let sc2_rns = &mut sc2.borrow_mut().rows;
    let sc2_cns = &mut sc2.borrow_mut().cols;
    sc1_rns.sort();
    sc1_cns.sort();
    sc2_rns.sort();
    sc2_cns.sort();
    let mut common_rows = {
        let mut x = Vec::new();
        let mut j = 0;
        for i in 0..sc1_rns.len() {
            let rn1 = sc1_rns[i];
            let mut rn2 = sc2_rns[j];
            while rn2 < rn1 {
                j += 1;
                rn2 = sc2_rns[j]
            }
            if rn1 == rn2 {
                x.push(rn1)
            }
        }
        x
    };
    let mut common_cols = {
        let mut x = Vec::new();
        let mut j = 0;
        for i in 0..sc1_cns.len() {
            let cn1 = sc1_cns[i];
            let mut cn2 = sc2_cns[j];
            while cn2 < cn1 {
                j += 1;
                cn2 = sc2_cns[j]
            }
            if cn1 == cn2 {
                x.push(cn1)
            }
        }
        x
    };

    while common_rows.len() > 0 && common_cols.len() > 0 {
        if sc2_rns.len() < sc2_cns.len() {
            let min_col = sc2_cns[0];
            let max_col = sc2_cns[sc2_cns.len() - 1];
            let common_max_pos = common_cols.binary_search(&max_col);
            let common_min_pos = common_cols.binary_search(&min_col);
            if common_max_pos.is_ok() {
                common_cols.remove(common_max_pos.unwrap());
                sc2_cns.pop();
            } else if common_min_pos.is_ok() {
                common_cols.remove(common_min_pos.unwrap());
                sc2_cns.remove(0);
            } else {
                common_cols.clear();
                sc2_cns.clear();
            }
        } else {
            let min_row = sc2_rns[0];
            let max_row = sc2_rns[sc2_rns.len() - 1];
            let common_max_pos = common_rows.binary_search(&max_row);
            let common_min_pos = common_rows.binary_search(&min_row);
            if common_max_pos.is_ok() {
                common_rows.remove(common_max_pos.unwrap());
                sc2_rns.pop();
            } else if common_min_pos.is_ok() {
                common_rows.remove(common_min_pos.unwrap());
                sc2_rns.remove(0);
            } else {
                common_rows.clear();
                sc2_rns.clear();
            }
        }
    }
}

fn align_supercells(
    supercells: &mut Vec<RefCell<TableObject>>,
    rows: &mut Vec<RefCell<TableObject>>,
    cols: &mut Vec<RefCell<TableObject>>,
    header_row_nums: Vec<usize>,
) -> Vec<RefCell<TableCell>> {
    // For each supercell, align it to the rows it intersects 50% of the height of,
    // and the columns it intersects 50% of the width of.
    // Eliminate supercells for which there are no rows and columns it intersects 50% with.
    let mut aligned_supercells = Vec::new();

    for supercell in supercells.iter() {
        let mut header = false;
        let mut row_bbox_rect: Option<BBox> = None;
        let mut col_bbox_rect: Option<BBox> = None;
        let mut header_intersects = HashSet::new();
        let mut data_intersects = HashSet::new();

        for i in 0..rows.len() {
            let row = &rows[i];
            let rh = row.borrow().bbox.y2 - row.borrow().bbox.y1;
            let sch = supercell.borrow().bbox.y2 - supercell.borrow().bbox.y1;
            let min_row_overlap = row.borrow().bbox.y1.min(supercell.borrow().bbox.y1);
            let max_row_overlap = row.borrow().bbox.y2.max(supercell.borrow().bbox.y2);
            let overlap_height = max_row_overlap - min_row_overlap;
            // python: if "span" in supercell do something else
            // - I couldn't find a place where we set the span key
            let overlap_fraction = overlap_height / rh;
            if overlap_fraction >= 0.5 {
                if header_row_nums.binary_search(&i).is_ok() {
                    header_intersects.insert(i);
                } else {
                    data_intersects.insert(i);
                }
            }
        }
        // Supercell cannot span across header boundary; eliminate whichever
        // is smallest
        if header_intersects.len() > 0 && data_intersects.len() > 0 {
            if header_intersects.len() > data_intersects.len() {
                data_intersects.clear();
            } else {
                header_intersects.clear();
            }
        }
        if header_intersects.len() > 0 {
            header = true;
        }
        // python: elif "span" in supercell continue
        // - see previous note about the "span" key
        let intersecting_rows = if header_intersects.len() > 0 {
            header_intersects
        } else {
            data_intersects
        };
        for rn in intersecting_rows.iter() {
            row_bbox_rect = match row_bbox_rect {
                Some(bb) => Some(bb.union(&rows[*rn].borrow().bbox)),
                None => Some(rows[*rn].borrow().bbox),
            }
        }
        if row_bbox_rect.is_none() {
            continue;
        }
        let row_bbox_rect = row_bbox_rect.unwrap();

        let mut intersecting_cols = Vec::new();
        for i in 0..cols.len() {
            let col = &cols[i];
            let cw = col.borrow().bbox.x2 - col.borrow().bbox.x1;
            let scw = supercell.borrow().bbox.x2 - supercell.borrow().bbox.x1;
            let min_col_overlap = col.borrow().bbox.x1.max(supercell.borrow().bbox.x1);
            let max_col_overlap = col.borrow().bbox.x2.min(supercell.borrow().bbox.x2);
            let overlap_width = max_col_overlap - min_col_overlap;
            // python: if "span" in supercell do something else
            // - couldn't find where we set the "span" key so drop that branch
            let overlap_fraction = overlap_width / cw;
            if overlap_fraction >= 0.5 {
                intersecting_cols.push(i);
            }
            col_bbox_rect = match col_bbox_rect {
                Some(bb) => Some(bb.union(&col.borrow().bbox)),
                None => Some(col.borrow().bbox),
            };
        }
        if col_bbox_rect.is_none() {
            continue;
        }
        let col_bbox_rect = col_bbox_rect.unwrap();

        supercell.borrow_mut().bbox = row_bbox_rect.intersect(&col_bbox_rect);

        // only a true supercell if it joins across multiple rows or columns
        if intersecting_cols.len() * intersecting_rows.len() > 1 {
            let row_numbers: Vec<usize> = intersecting_rows.into_iter().collect();
            aligned_supercells.push(RefCell::new(TableCell {
                rows: row_numbers,
                cols: intersecting_cols,
                inner: supercell.to_owned(),
            }));
            // python: if "span" in supercell and other stuff do other stuff
            // - see other span stuff.
        }
    }
    return aligned_supercells;
}

struct TableCell {
    rows: Vec<usize>,
    cols: Vec<usize>,
    inner: RefCell<TableObject>,
}

fn align_headers(
    col_headers: &mut Vec<RefCell<TableObject>>,
    rows: &mut Vec<RefCell<TableObject>>,
) -> Vec<usize> {
    let mut aligned_headers = Vec::new();
    let mut are_headers: Vec<bool> = Vec::with_capacity(rows.len());
    are_headers.fill(false);
    let mut header_row_nums: Vec<usize> = Vec::new();
    for header in col_headers.iter() {
        for i in 0..rows.len() {
            let r = &rows[i];
            let rh = r.borrow().bbox.y2 - r.borrow().bbox.y1;
            let min_row_overlap = r.borrow().bbox.y1.min(header.borrow().bbox.y1);
            let max_row_overlap = r.borrow().bbox.y2.max(header.borrow().bbox.y2);
            let overlap_height = max_row_overlap - min_row_overlap;
            if rh == 0.0 {
                if overlap_height == header.borrow().bbox.y2 - header.borrow().bbox.y1 {
                    header_row_nums.push(i);
                }
                continue;
            }
            if overlap_height / rh >= 0.5 {
                header_row_nums.push(i)
            }
        }
    }
    if header_row_nums.len() == 0 {
        return header_row_nums;
    }
    let mut header_rect: Option<BBox> = None;
    if header_row_nums[0] > 0 {
        // python: header_row_nums = list(range(header_row_nums[0] + 1)) + header_row_nums
        let first = header_row_nums[0];
        let origlen = header_row_nums.len();
        let mut temp: Vec<usize> = (0..first).collect();
        header_row_nums.append(&mut temp);
        for i in 0..first {
            header_row_nums.swap(i, i + origlen);
        }
    }

    for i in 0..header_row_nums.len() {
        // python: last = -1; for rn in header_rns: if rn == last + 1:
        // rust - usize can't be negative. point is to make sure nums are contiguous.
        let rn = header_row_nums[i];
        if rn == i {
            let row = &rows[rn];
            are_headers[rn] = true;
            header_rect = match header_rect {
                Some(bb) => Some(bb.union(&row.borrow().bbox)),
                None => Some(row.borrow().bbox),
            };
        } else {
            break;
        }
    }

    let header_rect = header_rect.unwrap();
    aligned_headers.push(RefCell::new(TableObject {
        label: Label::ColumnHeader,
        score: 1.00,
        bbox: header_rect,
    }));

    col_headers.clear();
    col_headers.append(&mut aligned_headers);
    header_row_nums
}

fn apply_threshold(objects: &mut Vec<RefCell<TableObject>>, threshold: f64) {
    for i in (0..objects.len()).rev() {
        if objects[i].borrow().score < threshold {
            objects.remove(i);
        }
    }
}

fn align_columns(columns: &mut Vec<RefCell<TableObject>>, rc_bbox: &BBox) {
    for c in columns {
        c.borrow_mut().bbox.y1 = rc_bbox.y1;
        c.borrow_mut().bbox.y2 = rc_bbox.y2;
    }
}

fn align_rows(rows: &mut Vec<RefCell<TableObject>>, rc_bbox: &BBox) {
    for r in rows {
        r.borrow_mut().bbox.x1 = rc_bbox.x1;
        r.borrow_mut().bbox.x2 = rc_bbox.x2;
    }
}

fn refine_rows(
    mut rows: &mut Vec<RefCell<TableObject>>,
    tokens: &Vec<&TableToken>,
    threshold: f64,
) {
    if tokens.len() > 0 {
        nms_by_containment(&mut rows, tokens, 0.5, true);
        remove_objects_without_content(tokens, &mut rows);
    } else {
        nms(&mut rows, NMSMatchCriteria::Object2Overlap, 0.5, true);
    }
    if rows.len() > 1 {
        rows.sort_by(|ob1, ob2| {
            (ob1.borrow().bbox.y1 + ob1.borrow().bbox.y2)
                .total_cmp(&(ob2.borrow().bbox.y1 + ob2.borrow().bbox.y2))
        });
    }
}

fn refine_columns(
    mut cols: &mut Vec<RefCell<TableObject>>,
    tokens: &Vec<&TableToken>,
    threshold: f64,
) {
    if tokens.len() > 0 {
        nms_by_containment(cols, tokens, 0.5, false);
        remove_objects_without_content(tokens, cols);
    } else {
        nms(&mut cols, NMSMatchCriteria::Object2Overlap, 0.5, true);
    }
    if cols.len() > 1 {
        cols.sort_by(|ob1, ob2| {
            (ob1.borrow().bbox.x1 + ob1.borrow().bbox.x2)
                .total_cmp(&(ob2.borrow().bbox.x1 + ob2.borrow().bbox.x2))
        });
    }
}

enum NMSMatchCriteria {
    Object2Overlap,
    Object1Overlap,
    IOU,
}

fn nms(
    objects: &mut Vec<RefCell<TableObject>>,
    match_criteria: NMSMatchCriteria,
    match_threshold: f32,
    keep_higher: bool,
) {
    if objects.len() == 0 {
        return;
    }

    objects.sort_by(|ob1, ob2| match keep_higher {
        true => (ob1.borrow().score * -1.0).total_cmp(&(ob2.borrow().score * -1.0)),
        false => ob1.borrow().score.total_cmp(&ob2.borrow().score),
    });

    let num_objects = objects.len();
    let mut suppression = Vec::with_capacity(num_objects);
    for _ in 0..num_objects {
        suppression.push(false);
    }

    for object2_num in 1..num_objects {
        let ob2_rect = &objects[object2_num].borrow().bbox;
        let ob2_area = ob2_rect.area();
        for object1_num in 0..object2_num {
            if !suppression[object1_num] {
                let ob1_rect = &objects[object1_num].borrow().bbox;
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

    for i in (0..objects.len()).rev() {
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

fn remove_objects_without_content(
    page_spans: &Vec<&TableToken>,
    objects: &mut Vec<RefCell<TableObject>>,
) {
    for i in (0..objects.len()).rev() {
        let obj = &mut objects[i];
        let (ob_text, _) = extract_text_inside_bbox(page_spans, &obj.borrow().bbox);
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
    container_objects: &mut Vec<RefCell<TableObject>>,
    package_objects: &Vec<&TableToken>,
    overlap_threshold: f32,
    _early_exit_vertical: bool,
) {
    container_objects
        .sort_by(|toa, tob| (toa.borrow().score * -1.0).total_cmp(&(tob.borrow().score * -1.0)));
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
struct MatchScore {
    pub container: RefCell<TableObject>,
    pub container_num: usize,
    pub score: f32,
}

fn slot_into_containers(
    container_objects: &Vec<RefCell<TableObject>>,
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

    let sorted_co = if _early_exit_vertical {
        let mut x = container_objects
            .iter()
            .enumerate()
            .map(|(i, ob)| (i, ob.clone()))
            .collect::<Vec<(usize, RefCell<TableObject>)>>();
        x.sort_by(|(_, toa), (_, tob)| toa.borrow().bbox.y1.total_cmp(&tob.borrow().bbox.y1));
        x
    } else {
        let mut x = container_objects
            .iter()
            .enumerate()
            .map(|(i, ob)| (i, ob.clone()))
            .collect::<Vec<(usize, RefCell<TableObject>)>>();
        x.sort_by(|(_, toa), (_, tob)| toa.borrow().bbox.x1.total_cmp(&tob.borrow().bbox.x1));
        x
    };

    for (pi, package) in package_objects.iter().enumerate() {
        let mut match_scores = Vec::new();
        let package_rect = &package.bbox;
        if package_rect.is_empty() {
            continue;
        }
        let package_area = package_rect.area();
        for (ci, container) in &sorted_co {
            if !_early_exit_vertical && container.borrow().bbox.x1 > container.borrow().bbox.x2 {
                if match_scores.len() == 0 {
                    match_scores.push(MatchScore {
                        container: container.clone(),
                        container_num: *ci,
                        score: 0.0,
                    });
                }
                break;
            } else if _early_exit_vertical
                && container.borrow().bbox.y1 > container.borrow().bbox.y2
            {
                if match_scores.len() == 0 {
                    match_scores.push(MatchScore {
                        container: container.clone(),
                        container_num: *ci,
                        score: 0.0,
                    });
                }
                break;
            }
            let container_rect = &container.borrow().bbox;
            let intersect_area = container_rect.intersect(package_rect).area();
            let overlap_fraction = intersect_area / package_area;
            match_scores.push(MatchScore {
                container: container.clone(),
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

    fn union(self: &Self, other: &Self) -> Self {
        let x1 = if self.x1 < other.x1 {
            other.x1
        } else {
            self.x1
        };
        let x2 = if self.x2 > other.x2 {
            other.x2
        } else {
            self.x2
        };
        let y1 = if self.y1 < other.y1 {
            other.y1
        } else {
            self.y2
        };
        let y2 = if self.y2 > other.y2 {
            other.y2
        } else {
            self.y2
        };

        if x1 >= x2 || y1 >= y2 {
            return BBox::empty();
        }
        return BBox { x1, y1, x2, y2 };
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
