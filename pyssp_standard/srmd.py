import os
import pathlib
import hashlib

from lxml import etree as et
from lxml.etree import QName

from pyssp_standard.utils import ModelicaXMLFile
from pyssp_standard.traceability import Classification, classification_parser_for, GElementCommon


class SRMD(ModelicaXMLFile):

    def __init__(self, file_path, mode='r'):
        self.name = os.path.basename(file_path)
        self.data = None
        self.checksum = None
        self.checksum_type = "SHA3-256"
        self.version = "1.0.0-beta2"
        self.common = GElementCommon()

        super().__init__(file_path, mode, "srmd11")

    def assign_data(self, filepath, create_checksum=True):
        if type(filepath) is not pathlib.PosixPath:
            filepath = pathlib.Path(filepath)
        self.data = str(filepath)

        if create_checksum:
            with open(filepath) as file:
                data = file.read()
                self.checksum = hashlib.sha3_256(data.encode()).hexdigest()

    def __read__(self):
        tree = et.parse(str(self.file_path))
        self.root = tree.getroot()
        self.version = self.root.get('version')
        self.name = self.root.get('name')
        self.data = self.root.get('data')
        self.checksum = self.root.get('checksum')
        self.checksum_type = self.root.get('checksumType')

        self.top_level_metadata.update(self.root.attrib)
        self.base_element.update(self.root.attrib)

        self.common.__read__(self.root)

    def __write__(self):
        attributes = {'version': self.version, 'name': self.name}
        if self.data is not None:
            attributes['data'] = self.data
        if self.checksum is not None:
            attributes['checksum'] = self.checksum
            attributes['checksumType'] = self.checksum_type

        self.root = et.Element(QName(self.namespaces['srmd'], 'SimulationResourceMetaData'), attrib=attributes)
        self.root = self.top_level_metadata.update_root(self.root)
        self.root = self.base_element.update_root(self.root)

        self.common.update_element(self.root)

    @property
    def classifications(self):
        return self.common.classifications

    @classifications.setter
    def classifications(self, classifications):
        self.common.classifications = classifications

    def add_classification(self, classification):
        self.common.add_classification(classification)
