import pytest
from pathlib import Path
import hashlib
import shutil
import tempfile

from lxml import etree as et

from pyssp_standard import STMD, SSP


@pytest.fixture
def write_file():
    with tempfile.NamedTemporaryFile(delete_on_close=False) as f:
        f.close()
        yield f.name


@pytest.fixture
def append_file():
    source_file = Path('./pytest/doc/test_schema_validation.stmd')
    with tempfile.NamedTemporaryFile(delete_on_close=False) as f:
        f.close()
        shutil.copyfile(source_file, f.name)
        yield f.name


@pytest.fixture
def write_ssp():
    with tempfile.NamedTemporaryFile(delete_on_close=False) as f:
        f.close()

        yield f.name


def test_read_stmd():
    test_file = Path('./pytest/doc/test_schema_validation.stmd')
    with STMD(test_file, 'r') as file:
        file.__check_compliance__()


def test_write_stmd(append_file):
    with STMD(append_file, 'a') as file:
        file.__check_compliance__()


def test_add_to_ssp(write_ssp):
    with SSP(write_ssp) as ssp:
        with ssp.stmd as stmd:
            pass  # DerivationChain entry should be added automatically

    with SSP(write_ssp, readonly=True) as ssp:
        with ssp.stmd as stmd:
            assert len(stmd.general_information.derivation_chain == 1)
