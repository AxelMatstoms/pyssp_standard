import pytest
from pathlib import Path
import hashlib
import shutil

from lxml import etree as et

from pyssp_standard import stmd, STMD, Classification, ClassificationEntry, classification_parser


@pytest.fixture
def write_file():
    test_file = Path('./test.stmd')
    yield test_file
    if test_file.exists():
        test_file.unlink()


@pytest.fixture
def append_file():
    source_file = Path('./pytest/doc/test_schema_validation.stmd')
    test_file = Path('./test.stmd')
    shutil.copyfile(source_file, test_file)

    yield test_file
    if test_file.exists():
        test_file.unlink()


def test_read_stmd():
    test_file = Path('./pytest/doc/test_schema_validation.stmd')
    with STMD(test_file, 'r') as file:
        file.__check_compliance__()


def test_write_stmd(append_file):
    with STMD(append_file, 'a') as file:
        file.__check_compliance__()
