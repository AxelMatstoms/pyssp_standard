import dataclasses
import datetime
import uuid
import urllib
import os

from lxml import etree as et
from lxml.etree import QName

from pyssp_standard.utils import ModelicaXMLFile
from pyssp_standard.standard import ModelicaStandard
from pyssp_standard.common_content_ssc import BaseElement, TopLevelMetaData


def gen_guid() -> str:
    return str(uuid.uuid4())


class ClassificationEntry(BaseElement, ModelicaStandard):
    """SSP Traceability ClassificationEntry

    This class represents a generic ClassificationEntry. As the standard
    is very flexible in what the "value" of a classification entry can
    be, the aim is not to fit every usecase, but rather to expose all
    data required for most common usecases. Therefore, the support for
    values other than text is only what is necessary to read and write
    in a standard-compliant manner.
    """
    keyword: str
    type_: str
    link: str | None
    linked_type: str | None
    content: list[et._Element]
    text: str

    def __init__(self, keyword_or_element: str | et._Element,
                 type_="text/plain",
                 link=None,
                 linked_type=None,
                 content: list[et._Element] | None = None,
                 text: str = "",
                 **kwargs):
        """ Construct a classification entry.

        This constructor should be called with either an lxml Element
        or keyword parameters for creating a new Classification Entry.
        Passing keyword parameters when constructing from an Element
        will result in values from kw parameters being overriden by the
        values in the XML element, or set to None if not present in the
        XML element.
        """
        super().__init__(**kwargs)

        self.content = []
        self.text = ""
        self.type_ = type_
        self.link = link
        self.linked_type = linked_type

        if isinstance(keyword_or_element, et._Element):
            self.content = []
            self.__read__(keyword_or_element)
        elif isinstance(keyword_or_element, str):
            self.keyword = keyword_or_element
            if content is not None:
                self.content = content
            self.text = text
        else:
            raise TypeError(f"Can't init ClassificationEntry with {type(keyword_or_element)}")

    def __read__(self, element):
        self.update({"id": element.get("id", ""), "description": element.get("description", "")})

        self.keyword = element.attrib["keyword"]
        self.type_ = element.get("type", "text/plain")
        self.link = element.get(QName(self.namespaces["xlink"], "href"))
        self.linked_type = element.get("linkedType")

        if element.text is not None:
            self.text = element.text

        for child in element:
            self.content.append(child)

    def __repr__(self):
        return f"ClassificationEntry(keyword={self.keyword}, text={self.text!r} content={self.content})"

    def as_element(self):
        entry = et.Element(QName(self.namespaces['stc'], 'ClassificationEntry'), attrib={'keyword': self.keyword})
        entry.text = self.text
        entry.extend(self.content)

        super().update_root(entry)

        if self.type_ != "text/plain":  # Only write if the value differs from the default
            entry.set("type", self.type_)

        if self.link is not None:  # optional
            entry.set(QName(self.namespaces["xlink"], "href"), self.link)

        if self.linked_type is not None:  # optional
            entry.set("linkedType", self.linked_type)

        return entry


class Classification(BaseElement, ModelicaStandard):
    """SSP Traceability Classification.

    This class represents a generic Classification. For easiser parsing
    of specific Classification types, this class can be subclassed, and
    registered with the @classification_parser decorator. A possible
    way to implement this is shown in the documentation for that
    decorator.
    """
    classification_type: str
    link: str
    linked_type: str
    classification_entries: list[ClassificationEntry]

    def __init__(self,
                 type_or_element: str | et._Element,
                 link: str | None = None,
                 linked_type: str | None = None,
                 entries: list[ClassificationEntry] | None = None,
                 **kwargs):
        """ Construct a classification.

        This constructor should be called with either an lxml Element
        or keyword parameters for creating a new Classification.
        Passing keyword parameters when constructing from an Element
        will result in values from kw parameters being overriden by the
        values in the XML element, or set to None if not present in the
        XML element.
        """
        super().__init__(**kwargs)

        self.link = link
        self.linked_type = linked_type
        self.classification_entries = []

        if isinstance(type_or_element, et._Element):
            self.__read__(type_or_element)
        elif isinstance(type_or_element, str):
            self.classification_type = type_or_element
            self.classification_entries = [] if entries is None else entries
        else:
            raise TypeError(f"Can't init Classification with {type(type_or_element)}")

    def add_classification_entry(self, classification_entry: ClassificationEntry):
        self.classification_entries.append(classification_entry)

    def __read__(self, element):
        # BaseElement doesn't work well with inheritance
        self.update({"id": element.get("id", ""), "description": element.get("description", "")})

        self.classification_type = element.attrib["type"]
        self.link = element.get(QName(self.namespaces["xlink"], "href"))
        self.linked_type = element.get("linkedType")

        for entry in element.getchildren():
            self.add_classification_entry(ClassificationEntry(entry))

    def as_element(self):
        classification = et.Element(QName(self.namespaces['stc'], 'Classification'),
                                    attrib={'type': self.classification_type})

        super().update_root(classification)

        if self.link is not None:
            classification.set(QName(self.namespaces["xlink"], "href"), self.link)

        if self.linked_type is not None:
            classification.set("linkedType", self.link)

        for entry in self.classification_entries:
            classification.append(entry.as_element())

        return classification


classification_parsers: dict[str, Classification] = {}


def classification_parser(type_: str):
    """Decorator for registering a classification parser for a given type.

    The parser class' constructor will be called with the XML element as
    its only argument. The class is expected to have a method as_element(self)
    that returns the XML element representation of the parsed classification.

    One possible implementation would be something like the following:

    >>> @classification_parser("com.example.custom")
    >>> class CustomClassification(Classification):
    >>>     test1: str
    >>>     test2: str

    >>>     def __init__(self, element=None, test1="", test2="", **kwargs):
    >>>         if element is not None:
    >>>             super().__init__(element)

    >>>             for entry in self.classification_entries:
    >>>                 if entry.keyword == "test1":
    >>>                     self.test1 = entry.text
    >>>                 elif entry.keyword == "test2":
    >>>                     self.test2 = entry.text
    >>>         else:
    >>>             super().__init__("com.example.custom", **kwargs)
    >>>             self.test1 = test1
    >>>             self.test2 = test2

    >>>     def as_element(self):
    >>>         self.classification_entries = [
    >>>                 ClassificationEntry("test1", text=self.test1),
    >>>                 ClassificationEntry("test2", text=self.test2)
    >>>         ]

    >>>         return super().as_element()
    """
    def decorator(parser):
        register_classification_parser(type_, parser)

        return parser

    return decorator


def classification_parser_for(type_: str) -> type:
    return classification_parsers.get(type_, Classification)


def register_classification_parser(type_: str, parser: type):
    classification_parsers[type_] = parser


class Annotations(ModelicaStandard):
    """ Stub implementation. """
    def __init__(self, element):
        self.element = element

    def as_element(self):
        return self.element


class GElementCommon(ModelicaStandard):
    classifications: list[Classification]
    annotations: Annotations | None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.classifications = []
        self.annotations = None

    def __read__(self, element: et._Element):
        classifications = element.findall('stc:Classification', self.namespaces)
        for classification in classifications:
            type_ = classification.attrib['type']
            cls = classification_parser_for(type_)  # select an appropriate parser based on the registry
            self.classifications.append(cls(classification))

        annotations = element.find("ssc:Annotations", self.namespaces)
        if annotations is not None:
            self.annotations = Annotations(annotations)

    def update_element(self, element: et.ElementBase):
        for classification in self.classifications:
            element.append(classification.as_element())

        if self.annotations is not None:
            element.append(self.annotations.as_element())

    def add_classification(self, classification: Classification):
        self.classifications.append(classification)


class DerivationChainEntry(BaseElement, TopLevelMetaData, ModelicaStandard):
    guid: str

    _BASE_ELEM_KEYS = {field.name for field in dataclasses.fields(BaseElement)}
    _TLMETA_KEYS = {field.name for field in dataclasses.fields(TopLevelMetaData)}

    def __init__(self, element=None, guid=None, **kwargs):
        if element is not None:
            self.__read__(element)
        else:
            # Ugly!
            BaseElement.__init__(self, **{k: w for k, w in kwargs.items() if k in self._BASE_ELEM_KEYS})
            TopLevelMetaData.__init__(self, **{k: w for k, w in kwargs.items() if k in self._TLMETA_KEYS})

            self.guid = gen_guid()

    def __read__(self, element):
        attrs = element.attrib
        BaseElement.update(self, {k: w for k, w in attrs.items() if k in self._BASE_ELEM_KEYS})
        TopLevelMetaData.update(self, {k: w for k, w in attrs.items() if k in self._TLMETA_KEYS})

        self.guid = element.get("GUID")

    def as_element(self):
        element = et.Element(QName(self.namespaces["stc"], "DerivationChainEntry"))

        if self.guid is not None:
            element.set("GUID", self.guid)

        # Ugly!! BaseElement and TopLevelMetaData do not inherit cooperatively
        for key in (self._BASE_ELEM_KEYS | self._TLMETA_KEYS):
            value = getattr(self, key)
            if value is not None and value != "":
                element.set(key, value)

        return element


class DerivationChain(BaseElement, ModelicaStandard):
    def __init__(self, element=None, entries=None, **kwargs):
        if element is not None:
            self.__read__(element)
        else:
            super().__init__(**kwargs)  # Init BaseElement with optional id, description
            self.entries = entries if entries is not None else []

    def __read__(self, element):
        self.update({"id": element.get("id", ""), "description": element.get("description", "")})
        self.entries = [DerivationChainEntry(child) for child in element]

    def add_chain_entry(self, **kwargs):
        """ Add a DerivationChainEntry.

        This method should be called before write, if modifications
        have occurred.
        """
        self.entries.append(DerivationChainEntry(**kwargs))

    def as_element(self):
        element = et.Element(QName(self.namespaces["stc"], "DerivationChain"))
        self.update_root(element)

        for entry in self.entries:
            element.append(entry.as_element())

        return element


class Links(ModelicaStandard):
    """ Stub implementation. """
    def __init__(self, element):
        self.element = element

    def as_element(self):
        return self.element


class Content(BaseElement, ModelicaStandard):
    text: str
    content: list[et.ElementBase]

    def __init__(self, element=None, text="", content=None, **kwargs):
        super().__init__(self, **kwargs)

        if element is not None:
            self.__read__(element)
        else:
            self.text = text
            self.content = content if content is not None else []

    def __read__(self, element):
        self.update({"id": element.get("id", ""), "description": element.get("description", "")})

        if element.text is not None:
            self.text = element.text

        self.content = [child for child in element]

    def as_element(self):
        element = et.Element(QName(self.namespaces["stc"], "Content"))
        self.update_root(element)

        element.text = self.text
        for elem in self.content:
            element.append(elem)

        return elem


class Signature(GElementCommon, BaseElement, ModelicaStandard):
    role: str
    type: str
    source: str | None
    source_base: str
    content: Content | None

    def __init__(
            self,
            element=None,
            role=None,
            type=None,
            source=None,
            source_base="file",
            content=None,
            signatures=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if element is not None:
            self.__read__(element)
        else:
            self.role = role
            self.type = type
            self.source = source
            self.source_base = source_base
            self.content = content
            self.signatures = signatures if signatures is not None else []

    def __read__(self, element):
        super().__read__(element)
        self.update({"id": element.get("id", ""), "description": element.get("description", "")})

        self.role = element.get("role")
        self.type = element.get("type")
        self.source = element.get("source")
        self.source_base = element.get("source_base")

        if (content := element.find("stc:Content", self.namespaces)) is not None:
            self.content = Content(content)

        self.signatures = [Signature(sig) for sig in element.findall("stc:Signature", self.namespaces)]

    def as_element(self):
        element = et.Element(QName(self.namespaces["stc"], "Signature"))
        self.update_root(element)  # Add BaseElement attributes

        element.set("role", self.role)
        element.set("type", self.type)

        if self.source is not None:
            element.set("source", self.source)

        if self.source_base != "file":
            element.set("sourceBase", self.source_base)

        if self.content is not None:
            element.append(self.content.as_element())

        for sig in self.signatures:
            element.append(sig.as_element())

        self.update_element(element)  # Add children from GElementCommon


class Summary(GElementCommon, BaseElement, ModelicaStandard):
    type: str
    source: str | None
    source_base: str
    content: Content | None
    signatures: list[Signature]

    def __init__(
            self,
            element=None,
            type=None,
            source=None,
            source_base="file",
            content=None,
            signatures=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if element is not None:
            self.__read__(element)
        else:
            self.type = type
            self.source = source
            self.source_base = source_base
            self.content = content
            self.signatures = signatures if signatures is not None else []

    def __read__(self, element):
        super().__read__(element)
        self.update({"id": element.get("id", ""), "description": element.get("description", "")})

        self.type = element.get("type")
        self.source = element.get("source")
        self.source_base = element.get("sourceBase")

        if (content := element.find("stc:Content", self.namespaces)) is not None:
            self.content = Content(content)

        self.signatures = [Signature(sig) for sig in element.findall("stc:Signature", self.namespaces)]

    def as_element(self):
        element = et.Element(QName(self.namespaces["stc"], "Summary"))
        self.update_root(element)  # Add BaseElement attributes

        # Mandatory attributes shouldn't be None anyway
        element.set("type", self.type)

        if self.source is not None:
            element.set("source", self.source)

        if self.source_base != "file":
            element.set("sourceBase", self.source_base)

        if self.content is not None:
            element.append(self.content.as_element())

        for sig in self.signatures:
            element.append(sig.as_element())

        self.update_element(element)  # Add children from GElementCommon


class MetaData(GElementCommon, BaseElement, ModelicaStandard):
    kind: str
    type: str
    source: str | None
    source_base: str
    content: Content | None
    signatures: list[Signature]

    def __init__(
            self,
            element=None,
            kind=None,
            type=None,
            source=None,
            source_base="file",
            content=None,
            signatures=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if element is not None:
            self.__read__(element)
        else:
            self.kind = kind
            self.type = type
            self.source = source
            self.source_base = source_base
            self.content = content
            self.signatures = signatures if signatures is not None else []

    def __read__(self, element):
        super().__read__(element)
        self.update({"id": element.get("id", ""), "description": element.get("description", "")})

        self.kind = element.get("kind")
        self.type = element.get("type")
        self.source = element.get("source")
        self.source_base = element.get("source_base")

        if (content := element.find("stc:Content", self.namespaces)) is not None:
            self.content = Content(content)

        self.signatures = [Signature(sig) for sig in element.findall("stc:Signature", self.namespaces)]

    def as_element(self):
        element = et.Element(QName(self.namespaces["stc"], "MetaData"))
        self.update_root(element)  # Add BaseElement attributes

        element.set("kind", self.kind)
        element.set("type", self.type)

        if self.source is not None:
            element.set("source", self.source)

        if self.source_base != "file":
            element.set("sourceBase", self.source_base)

        if self.content is not None:
            element.append(self.content.as_element())

        for sig in self.signatures:
            element.append(sig.as_element())

        self.update_element(element)  # Add children from GElementCommon


class Resource(GElementCommon, BaseElement, ModelicaStandard):
    resource_manager: "ResourceManager"

    kind: str
    scope: str | None
    type: str
    source: str | None
    master: str | None

    content: Content | None
    summary: Summary | None
    metadata: list[MetaData]
    signatures: list[Signature]

    def __init__(
            self,
            element=None,
            resource_manager=None,
            kind=None,
            scope=None,
            type=None,
            source=None,
            master=None,
            content=None,
            summary=None,
            metadata=None,
            signatures=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.resource_manager = resource_manager

        self.kind = kind
        self.scope = scope
        self.type = type
        self.source = source
        self.master = master

        self.content = content
        self.summary = summary
        self.metadata = metadata if metadata is not None else []
        self.signatures = signatures if signatures is not None else []

        if element is not None:
            self.__read__(element, resource_manager)

    def __read__(self, element, resource_manager):
        super().__read__(element)
        self.update({"id": element.get("id", ""), "description": element.get("description", "")})

        self.kind = element.get("kind")
        self.scope = element.get("scope")
        self.type = element.get("type")
        self.source = element.get("source")
        self.master = element.get("master")

        if (content := element.find("stc:Content", self.namespaces)) is not None:
            self.content = Content(content)

        if (summary := element.find("stc:Summary", self.namespaces)) is not None:
            self.summary = Summary(content)

        self.metadata = [MetaData(elem) for elem in element.findall("stc:MetaData", self.namespaces)]
        self.signatures = [Signature(elem) for elem in element.findall("stc:Signature", self.namespaces)]

    def as_element(self):
        element = et.Element(QName(self.namespaces["stc"], "Resource"))
        self.update_root(element)  # Add BaseElement attributes

        if self.kind is not None:
            element.set("kind", self.kind)

        if self.scope is not None:
            element.set("scope", self.scope)

        if self.type is not None:
            element.set("type", self.type)

        if self.source is not None:
            element.set("source", self.source)

        if self.master is not None:
            element.set("master", self.master)

        if self.content is not None:
            element.append(self.content.as_element())

        if self.summary is not None:
            element.append(self.summary.as_element())

        for metadata in self.metadata:
            element.append(metadata.as_element())

        for signature in self.signatures:
            element.append(signature.as_element())

        self.update_element(element)  # Add children from GElementCommon

        return element

    def as_reference(self):
        return ResourceReference(resource_manager=self.resource_manager, ref_id=self.id)


class ResourceReference(ModelicaStandard):
    """ Reference to a Resource.

    A ResourceReference is a proxy to to the Resource it references.
    The Resources attributes may be accessed through the properties
    defines in this class.
    """
    ref_id: str
    resource_manager: "ResourceManager"

    def __init__(self, element=None, resource_manager=None, ref_id=None):
        self.resource_manager = resource_manager
        if element is not None:
            self.__read__(element)
        else:
            self.ref_id = ref_id

    def __read__(self, element):
        uri = element.get(QName(self.namespaces["xlink"], "href"))
        if uri is not None:
            split = urllib.parse.urldefrag(uri)

            assert split.url == ""

            self.ref_id = split.fragment
        else:
            self.ref_id = None

    def as_element(self):
        element = et.Element(QName(self.namespaces["stc"], "ResourceReference"))
        element.set(QName(self.namespaces["xlink"], "href"), self.ref_id)

        return element

    @property
    def kind(self) -> str:
        return self.resource_manager[self.ref_id].kind

    @kind.setter
    def kind(self, kind) -> str:
        self.resource_manager[self.ref_id].kind = kind

    @property
    def scope(self, ) -> str:
        return self.resource_manager[self.ref_id].scope

    @scope.setter
    def scope(self, scope) -> str:
        self.resource_manager[self.ref_id].scope = scope

    @property
    def type(self) -> str:
        return self.resource_manager[self.ref_id].type

    @type.setter
    def type(self, type) -> str:
        self.resource_manager[self.ref_id].type = type

    @property
    def source(self) -> str:
        return self.resource_manager[self.ref_id].source

    @source.setter
    def source(self, source) -> str:
        self.resource_manager[self.ref_id].source = source

    @property
    def master(self) -> str:
        return self.resource_manager[self.ref_id].master

    @master.setter
    def master(self, master) -> str:
        self.resource_manager[self.ref_id].master = master

    @property
    def content(self) -> str:
        return self.resource_manager[self.ref_id].content

    @content.setter
    def content(self, content) -> str:
        self.resource_manager[self.ref_id].content = content

    @property
    def summary(self) -> str:
        return self.resource_manager[self.ref_id].summary

    @summary.setter
    def summary(self, summary) -> str:
        self.resource_manager[self.ref_id].summary = summary

    @property
    def metadata(self) -> str:
        return self.resource_manager[self.ref_id].type

    @metadata.setter
    def metadata(self, metadata) -> str:
        self.resource_manager[self.ref_id].type = metadata

    @property
    def signatures(self) -> str:
        return self.resource_manager[self.ref_id].signatures

    @signatures.setter
    def signatures(self, signatures) -> str:
        self.resource_manager[self.ref_id].signatures = signatures


class GeneralInformation(GElementCommon, BaseElement, ModelicaStandard):
    derivation_chain: DerivationChain | None
    resources: list[Resource]
    links: Links | None

    ns: str

    def __init__(
            self,
            element=None,
            resource_manager=None,
            ns=None,
            derivation_chain=None,
            resources=None,
            links=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.ns = ns
        self.derivation_chain = derivation_chain
        self.resources = resources if resources is not None else []
        self.links = links

        if element is not None:
            self.__read__(element, resource_manager)

    def __read__(self, element, resource_manager):
        super().__read__(element)
        self.update({"id": element.get("id", ""), "description": element.get("description", "")})

        self.ns = QName(element.tag).namespace

        if (deriv_chain := element.find("stc:DerivationChain", self.namespaces)) is not None:
            self.derivation_chain = DerivationChain(deriv_chain)

        resources = element.xpath("stc:Resource|stc:ResourceReference", namespaces=self.namespaces)
        for elem in resources:
            if elem.tag == QName(self.namespaces["stc"], "Resource"):
                res = Resource(elem, resource_manager=resource_manager)
            else:
                res = ResourceReference(elem, resource_manager=resource_manager)

            self.resources.append(res)

        links = element.find('stc:Links', self.namespaces)
        if links is not None:
            self.links = Links(links)

    def as_element(self):
        element = et.Element(QName(self.ns, "GeneralInformation"))
        self.update_root(element)  # Add BaseElement attributes

        if self.derivation_chain is not None:
            element.append(self.derivation_chain.as_element())

        for resource in self.resources:
            element.append(resource.as_element())

        if self.links is not None:
            element.append(self.links.as_element())

        self.update_element(element)  # Add children from GElementCommon

        return element


class LifeCycleInformation(ModelicaStandard):
    """ Stub implementation. """
    def __init__(self, element):
        self.element = element

    def as_element(self):
        return self.element


class GPhaseCommon(GElementCommon, ModelicaStandard):
    links: Links | None
    lifecycle_information: LifeCycleInformation

    def __init__(self, links=None, lifecycle_information=None, **kwargs):
        super().__init__(**kwargs)

        self.links = links
        self.lifecycle_information = lifecycle_information

    def __read__(self, element):
        super().__read__(element)

        links = element.find('stc:Links', self.namespaces)
        if links is not None:
            self.links = Links(links)

        lifecycle_information = element.find('stc:LifeCycleInformation', self.namespaces)
        if lifecycle_information is not None:
            self.lifecycle_information = LifeCycleInformation(lifecycle_information)

    def update_element(self, element: et.ElementBase):
        if self.links is not None:
            element.append(self.links.as_element())

        if self.lifecycle_information is not None:
            element.append(self.lifecycle_information.as_element())

        super().update_element(element)


def snake_to_cap_words(snake):
    """Convert snake_case to CapWords."""
    return "".join(part.capitalize() for part in snake.split("_"))


def select(kwargs, keys):
    """Select a set of keys from a dictionary.

    Returns a tuple of selected, and remaining kv pairs.
    """
    selected = {k: w for k, w in kwargs.items() if k in keys}
    remaining = {k: w for k, w in kwargs.items() if k not in keys}

    return selected, remaining


def make_phase(phase_cls: type):
    """ Generate a Phase class.

    To avoid extensive code repetetion, use some code generation to
    generate automatically generate phases based on type annotations.
    """
    phase_name = phase_cls.__name__
    namespace = phase_cls.__module__.split(".")[-1]  # This is **really** stupid!
    step_names = list(phase_cls.__annotations__.keys())
    step_xml_names = [snake_to_cap_words(name) for name in step_names]

    def __init__(self, element=None, resource_manager=None, **kwargs):
        selected, remaining = select(kwargs, step_names)
        super(phase_cls, self).__init__(**remaining)

        for name in step_names:
            setattr(self, name, None)

        if element is not None:
            self.__read__(element, resource_manager)
        else:
            for name, step in selected.items():
                setattr(self, name, step)

    def __read__(self, element, resource_manager):
        super(phase_cls, self).__read__(element)

        for name, xml_name in zip(step_names, step_xml_names):
            if (step := element.find(f"{namespace}:{xml_name}", self.namespaces)) is not None:
                setattr(self, name, Step(step, resource_manager, tag_name=xml_name))

    def as_element(self):
        element = et.Element(QName(self.namespaces[namespace], phase_name))
        self.update_root(element)

        for name, xml_name in zip(step_names, step_xml_names):
            if (step := getattr(self, name)) is not None:
                element.append(step.as_element(QName(self.namespaces[namespace], xml_name)))

        self.update_element(element)

        return element

    phase_cls.__init__ = __init__
    phase_cls.__read__ = __read__
    phase_cls.as_element = as_element

    return phase_cls


class Particle(GElementCommon, BaseElement, ModelicaStandard):
    tag_name: str
    resources: list[Resource]

    def __init__(
            self,
            element=None,
            resource_manager=None,
            tag_name=None,
            resources=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.resources = resources if resources is not None else []

        if element is not None:
            self.__read__(element, resource_manager)
        else:
            assert tag_name is not None, "Particle must have a tag name"
            self.tag_name = tag_name

    def __read__(self, element, resource_manager):
        super().__read__(element)
        self.update({"id": element.get("id", ""), "description": element.get("description", "")})

        self.tag_name = element.tag

        resources = element.xpath("stc:Resource|stc:ResourceReference", namespaces=self.namespaces)
        for elem in resources:
            if elem.tag == QName(self.namespaces["stc"], "Resource"):
                res = Resource(elem, resource_manager=resource_manager)
            else:
                res = ResourceReference(elem, resource_manager=resource_manager)

            self.resources.append(res)

    def as_element(self):
        # Could be stc:Input, stc:Procedure, etc.
        element = et.Element(self.tag_name)
        self.update_root(element)  # Add BaseElement attributes

        for resource in self.resources:
            element.append(resource.as_element())

        self.update_element(element)  # Add children from GElementCommon

        return element


class Step(GElementCommon, BaseElement, ModelicaStandard):
    input: Particle
    procedure: Particle
    output: Particle
    rationale: Particle
    links: Links
    lifecycle_information: LifeCycleInformation

    def __init__(
            self,
            element=None,
            resource_manager=None,
            tag_name=None,
            input=None,
            procedure=None,
            output=None,
            rationale=None,
            links=None,
            lifecycle_information=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.input = input
        self.procedure = procedure
        self.output = output
        self.rationale = rationale
        self.links = links
        self.lifecycle_information = lifecycle_information

        if element is not None:
            self.__read__(element, resource_manager)

    def __read__(self, element, resource_manager):
        super().__read__(element)
        self.update({"id": element.get("id", ""), "description": element.get("description", "")})

        if (input_ := element.find("stc:Input", self.namespaces)) is not None:
            self.input = Particle(input_, resource_manager=resource_manager)

        if (procedure := element.find("stc:Procedure", self.namespaces)) is not None:
            self.procedure = Particle(procedure, resource_manager=resource_manager)

        if (output := element.find("stc:Output", self.namespaces)) is not None:
            self.output = Particle(output, resource_manager=resource_manager)

        if (rationale := element.find("stc:Rationale", self.namespaces)) is not None:
            self.rationale = Particle(rationale, resource_manager=resource_manager)

        if (links := element.find("stc:Links", self.namespaces)) is not None:
            self.links = Links(links)

        if (lifecycle_information := element.find("stc:LifeCycleInformation", self.namespaces)) is not None:
            self.lifecycle_information = LifeCycleInformation(lifecycle_information)

    def as_element(self, tag_name):
        # Could be stmd:VerifyAnalysis, or dtmd:VerifyTasks, ...
        element = et.Element(tag_name)
        self.update_root(element)  # Add BaseElement attributes

        if self.input is not None:
            element.append(self.input.as_element())

        if self.procedure is not None:
            element.append(self.procedure.as_element())

        if self.output is not None:
            element.append(self.output.as_element())

        if self.rationale is not None:
            element.append(self.rationale.as_element())

        if self.links is not None:
            element.append(self.links.as_element())

        if self.lifecycle_information is not None:
            element.append(self.lifecycle_information.as_element())

        self.update_element(element)  # Add children from GElementCommon

        return element


class ResourceManager:
    resources: dict[str, Resource]
    written: set[str]

    def __init__(self):
        self.resources = {}
        self.written = set()

    def add_resource(self, resource: Resource):
        if resource.id is None:
            # Generate a new id if the resource doesn't already have one
            resource.id = gen_guid()

        self.resources[resource.id] = resource

    def __getitem__(self, ref_id):
        return self.resources[ref_id]

    def as_element(self, resource):
        if resource.id is None:
            return resource.as_element()

        if resource.id not in self.written:
            self.written.add(resource.id)
            return resource.as_element()
        else:
            return resource.as_reference_element()


class TaskMetaData(ModelicaXMLFile):
    version: str
    name: str
    guid: str

    general_information: GeneralInformation
    common: GElementCommon

    def __init__(self, file_path, mode='r', identifier="unknown"):
        self.name = os.path.basename(file_path)
        self.version = "1.0.0-beta2"
        self.guid = gen_guid()

        self.resource_manager = ResourceManager()
        self.general_information = None
        self.common = GElementCommon()

        super().__init__(file_path, mode, identifier)

    def __read__(self):
        tree = et.parse(str(self.file_path))
        self.root = tree.getroot()

        if (version := self.root.get("version")) is not None:
            self.version = version

        if (name := self.root.get("name")) is not None:
            self.name = name

        if (guid := self.root.get("GUID")) is not None:
            self.guid = guid

        self.top_level_metadata.update(self.root.attrib)
        self.base_element.update(self.root.attrib)
        self.common.__read__(self.root)

        if (gen_info := self.root.find(f"{self.ns}:GeneralInformation", self.namespaces)) is not None:
            self.general_information = GeneralInformation(gen_info, self.resource_manager)

    def __write__(self):
        self.guid = gen_guid()
        self.add_derivation_chain_entry()
        self.top_level_metadata.generationDateAndTime = datetime.datetime.now().isoformat()
        self.top_level_metadata.generationTool = "pyssp_standard"

        attributes = {"version": self.version, "name": self.name, "GUID": self.guid}
        self.root = et.Element(
                QName(self.namespaces[self.ns], self.tag_name),
                attributes,
                nsmap=self.namespaces
        )

        self.top_level_metadata.update_root(self.root)
        self.base_element.update_root(self.root)

        self.root.append(self.general_information.as_element())

        # Intentionally don't write GElementCommon, since it should be last

    def add_derivation_chain_entry(self):
        if self.general_information is None:
            self.general_information = GeneralInformation()

        if self.general_information.derivation_chain is None:
            self.general_information.derivation_chain = DerivationChain()

        self.general_information.derivation_chain.add_chain_entry()

    @property
    def classifications(self):
        return self.common.classifications

    @classifications.setter
    def classifications(self, classifications):
        self.common.classifications = classifications

    def add_classification(self, classification):
        self.common.add_classification(classification)
