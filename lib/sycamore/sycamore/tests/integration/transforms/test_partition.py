import re
import string

import nltk

from sycamore.transforms.partition import SYCAMORE_DETR_MODEL, SycamorePartitioner
from sycamore.data import TableElement
from sycamore.data.table import Table, TableCell
import sycamore
from sycamore.tests.config import TEST_DIR

from sycamore.transforms.partition import ArynPartitioner
import os

MODEL_SERVER_KEY = os.environ["MODEL_SERVER_KEY"]


def test_detr_ocr():
    path = TEST_DIR / "resources/data/pdfs/Transformer.pdf"

    context = sycamore.init()

    # TODO: The title on the paper is recognized as a section header rather than a page header at the moment.
    # The test will need to be updated if and when that changes.
    docs = (
        context.read.binary(paths=[str(path)], binary_format="pdf")
        .partition(SycamorePartitioner(SYCAMORE_DETR_MODEL, use_ocr=True))
        .explode()
        .filter(lambda doc: "page_number" in doc.properties and doc.properties["page_number"] == 1)
        .filter(lambda doc: doc.type == "Section-header")
        .take_all()
    )

    assert "Attention Is All You Need" in set(str(d.text_representation).strip() for d in docs)


def check_table_extraction(**kwargs):
    path = TEST_DIR / "resources/data/pdfs/basic_table.pdf"

    basic_table_result = Table(
        [
            TableCell(content="Grade.", rows=[0, 1], cols=[0], is_header=True),
            TableCell(content="Yield Point.", rows=[0, 1], cols=[1], is_header=True),
            TableCell(content="Ultimate tensile strength", rows=[0], cols=[2, 3], is_header=True),
            TableCell(content="Per cent elong. 50.8 mm or 2 in.", rows=[0, 1], cols=[4], is_header=True),
            TableCell(content="Per cent reduct. area.", rows=[0, 1], cols=[5], is_header=True),
            TableCell(content="kg/mm2", rows=[1], cols=[2], is_header=True),
            TableCell(content="lb/in2", rows=[1], cols=[3], is_header=True),
            TableCell(content="Hard", rows=[2], cols=[0]),
            TableCell(content="0.45 ultimate", rows=[2], cols=[1]),
            TableCell(content="56.2", rows=[2], cols=[2]),
            TableCell(content="80,000", rows=[2], cols=[3]),
            TableCell(content="15", rows=[2], cols=[4]),
            TableCell(content="20", rows=[2], cols=[5]),
            TableCell(content="Medium", rows=[3], cols=[0]),
            TableCell(content="0.45 ultimate", rows=[3], cols=[1]),
            TableCell(content="49.2", rows=[3], cols=[2]),
            TableCell(content="70,000", rows=[3], cols=[3]),
            TableCell(content="18", rows=[3], cols=[4]),
            TableCell(content="25", rows=[3], cols=[5]),
            TableCell(content="Soft", rows=[4], cols=[0]),
            TableCell(content="0.45 ultimate", rows=[4], cols=[1]),
            TableCell(content="42.2", rows=[4], cols=[2]),
            TableCell(content="60,000", rows=[4], cols=[3]),
            TableCell(content="22", rows=[4], cols=[4]),
            TableCell(content="30", rows=[4], cols=[5]),
        ]
    )

    from sycamore.utils.time_trace import ray_logging_setup

    context = sycamore.init(ray_args={"runtime_env": {"worker_process_setup_hook": ray_logging_setup}})

    # TODO: The title on the paper is recognized as a section header rather than a page header at the moment.
    # The test will need to be updated if and when that changes.
    docs = (
        context.read.binary(paths=[str(path)], binary_format="pdf")
        .partition(SycamorePartitioner(extract_table_structure=True, **kwargs))
        .take_all()
    )

    assert len(docs) == 1
    doc = docs[0]
    tables = [e for e in doc.elements if e.type == "table"]
    assert len(tables) == 1
    assert isinstance(tables[0], TableElement)
    assert tables[0].table is not None

    for cell1, cell2 in zip(tables[0].table.cells, basic_table_result.cells):
        # Compare without bbox.

        # Comparing text is a bit tricky here. In a perfect world we could do exact matches, but
        # OCR has enough gotchas that that usually fails. Common differences include whitespace,
        # punctuation, and things like superscripts. We use Levenshtein distance as an alternative.

        res_content1 = re.sub(r"\s+", "", cell1.content.translate(str.maketrans("", "", string.punctuation)))
        res_content2 = re.sub(r"\s+", "", cell2.content.translate(str.maketrans("", "", string.punctuation)))
        # assert res_content1 == res_content2
        distance = nltk.edit_distance(res_content1, res_content2)

        print(f"edit distance: {distance}")

        assert distance <= 2

        assert cell1.rows == cell2.rows
        assert cell1.cols == cell2.cols
        assert cell1.is_header == cell2.is_header


def test_table_extraction_with_ocr():
    check_table_extraction(use_ocr=True)


def test_table_extraction_with_no_ocr():
    check_table_extraction(use_ocr=False)


def test_aryn_partitioner():
    path = TEST_DIR / "resources/data/pdfs/Transformer.pdf"

    context = sycamore.init()

    docs = (
        context.read.binary(paths=[str(path)], binary_format="pdf")
        .partition(ArynPartitioner(aryn_token=MODEL_SERVER_KEY))
        .explode()
        .filter(lambda doc: "page_number" in doc.properties and doc.properties["page_number"] == 1)
        .filter(lambda doc: doc.type == "Section-header")
        .take_all()
    )

    assert "Attention Is All You Need" in set(str(d.text_representation).strip() for d in docs)


def test_table_extraction_with_ocr_batched():
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-8s %(asctime)s   %(filename)s:%(lineno)d   %(message)s"
    )
    check_table_extraction(use_ocr=True, batch_at_a_time=True)


def test_sycamore_batched_sequenced():
    import pathlib
    from sycamore.transforms.detr_partitioner import SycamorePDFPartitioner
    from sycamore.tests.unit.transforms.compare_detr_impls import compare_batched_sequenced

    s = SycamorePDFPartitioner("Aryn/deformable-detr-DocLayNet")
    for pdf in pathlib.Path(TEST_DIR).rglob("*.pdf"):
        print(f"Testing {pdf}")
        p = compare_batched_sequenced(s, pdf)
        print(f"Compared {len(p)} pages")
