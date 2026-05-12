from dataclasses import dataclass, field

from lxml import etree as et
from lxml.etree import QName

from pyssp_standard.transformation_types import Transformation, TRANSFORMATION_CHOICE_XPATH
from pyssp_standard.common_content_ssc import Annotations, Annotation, BaseElement, TopLevelMetaData
from pyssp_standard.utils import ModelicaXMLFile, wraps_dataclass
from pyssp_standard.standard import ModelicaStandard


@dataclass
class MappingEntry(ModelicaStandard):
    source: str
    target: str
    suppress_unit_conversion: bool = False
    annotations: Annotations | None = None
    transformation: Transformation | None = None

    @classmethod
    def from_xml(cls, elem):
        source = elem.get("source")
        target = elem.get("target")
        suppress_unit_conversion = elem.get("suppressUnitConversion", False)

        transf_elem = elem.xpath(
            TRANSFORMATION_CHOICE_XPATH, namespaces={"ssc": cls.namespaces["ssc"]}
        )
        transf_elem = transf_elem[0] if transf_elem else None
        transformation = None
        if transf_elem is not None:
            transformation = Transformation(transformation=transf_elem)

        anno_list = None
        annotations = elem.find('ssc:Annotations', cls.namespaces)
        if annotations is not None:
            annotations_list = annotations.findall('ssc:Annotation', cls.namespaces)
            anno_list = Annotations()
            for anno in annotations_list:
                anno_item = Annotation(type_declaration=anno.get('type'))
                anno_item.add_element(anno)
                anno_list.add_annotation(anno_item)

        return cls(
            source=source,
            target=target,
            suppress_unit_conversion=suppress_unit_conversion,
            annotations=annotations,
            transformation=transformation
        )

    def to_xml(self):
        mapping_entry = et.Element(
            QName(self.namespaces["ssm"], "MappingEntry"),
            attrib={"target": self.target, "source": self.source},
        )

        if self.suppress_unit_conversion:
            mapping_entry.set("suppressUnitConversion", "true")

        if self.transformation is not None:
            transformation_element = self.transformation.element()
            if transformation_element is not None:
                mapping_entry.append(transformation_element)
        if self.annotations is not None and not self.annotations.is_empty():
            annotation_element = self.annotations.root
            if annotation_element is not None:
                mapping_entry.append(annotation_element)

        return mapping_entry


@dataclass
class SSMElem(ModelicaStandard):
    version: str = "1.0"
    entries: list[MappingEntry] = field(default_factory=list)
    base_element: BaseElement = field(default_factory=BaseElement)
    top_level_metadata: TopLevelMetaData = field(default_factory=TopLevelMetaData)

    @classmethod
    def from_xml(cls, elem):
        version = elem.get("version", "1.0")
        entries = [
            MappingEntry.from_xml(e) for e in elem.findall("ssm:MappingEntry", cls.namespaces)
        ]

        base_element = BaseElement()
        top_level_metadata = TopLevelMetaData()

        base_element.update(elem.attrib)
        top_level_metadata.update(elem.attrib)

        return cls(
            version=version,
            entries=entries,
            base_element=base_element,
            top_level_metadata=top_level_metadata,
        )

    def to_xml(self):
        namespaces = ["ssm", "ssc"]
        nsmap = {k: self.namespaces[k] for k in namespaces}
        root = et.Element(
            QName(self.namespaces["ssm"], "ParameterMapping"),
            nsmap=nsmap,
            attrib={"version": self.version}
        )
        self.base_element.update(root.attrib)
        self.top_level_metadata.update(root.attrib)

        for entry in self.entries:
            root.append(entry.to_xml())

        return root


@wraps_dataclass(SSMElem, local_name="ssm_elem")
class SSM(ModelicaXMLFile):
    def __init__(self, filepath, mode="r"):
        self.ssm_elem = SSMElem()
        super().__init__(file_path=filepath, mode=mode, identifier="ssm")

    def __read__(self):
        tree = et.parse(str(self.file_path))
        self.root = tree.getroot()
        self.ssm_elem = SSMElem.from_xml(self.root)

    def __write__(self):
        self.root = self.ssm_elem.to_xml()

    @property
    def identifier(self):
        if self.version == "2.0":
            return "ssm2"
        else:
            return "ssm"

    @property
    def mappings(self):
        return self.entries

    def add_mapping(
        self, source, target, suppress_unit_conversion=False, transformation=None, annotations=None
    ):
        self.entries.append(
            MappingEntry(
                source=source,
                target=target,
                suppress_unit_conversion=suppress_unit_conversion,
                transformation=transformation,
                annotations=annotations,
            )
        )

    def edit_mapping(self, edit_target=True, *, target=None, source=None,
                     transformation: Transformation = None, suppress_unit_conversion=None,
                     annotations: Annotations = None):
        found = False
        idx = 0
        for idx, entry in enumerate(self.entries):
            if edit_target and entry.target == target:
                found = True
                break
            elif not edit_target and entry.source == source:
                found = True
                break

        if found:
            mapping_found = self.entries[idx]
            if target is not None:
                mapping_found.target = target
            if source is not None:
                mapping_found.source = source
            if transformation is not None:
                mapping_found.transformation = transformation
            if suppress_unit_conversion is not None:
                mapping_found.suppress_unit_conversion = suppress_unit_conversion
            if annotations is not None:
                mapping_found.annotations = annotations

        else:
            raise Exception("The target or source was not found, there is nothing to edit")
