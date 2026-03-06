from typing import List, NewType, Protocol, Tuple, runtime_checkable


@runtime_checkable
class DoclingBBox(Protocol):
    """Structural interface for Docling BoundingBox objects."""

    l: float  # noqa: E741
    r: float
    t: float
    b: float


@runtime_checkable
class DoclingPageSize(Protocol):
    """Structural interface for Docling PageSize objects."""

    width: float
    height: float


@runtime_checkable
class DoclingProv(Protocol):
    """Structural interface for Docling ProvenanceItem objects."""

    page_no: int
    bbox: DoclingBBox


@runtime_checkable
class DoclingEntity(Protocol):
    """Structural interface for objects with provenance (TextItem, TableData)."""

    prov: List[DoclingProv]


@runtime_checkable
class DoclingText(Protocol):
    """Structural interface for Docling Text objects."""

    text: str
    self_ref: str
    prov: List[DoclingProv]


@runtime_checkable
class DoclingDoc(Protocol):
    """Structural interface for the DoclingDocument object."""

    groups: List['DoclingGroup']
    texts: List[DoclingText]
    body: 'DoclingGroup'


@runtime_checkable
class DoclingPage(Protocol):
    """Structural interface for Docling Page objects."""

    size: DoclingPageSize


@runtime_checkable
class DoclingTextRef(Protocol):
    """Structural interface for Docling Text reference objects."""

    def resolve(self, doc: DoclingDoc) -> DoclingText: ...


@runtime_checkable
class DoclingGroup(Protocol):
    """Structural interface for Docling Group objects."""

    label: str
    name: str
    children: List[DoclingTextRef]


TextID = NewType('TextID', int)
Ngram = NewType('Ngram', str)

# Common Mapping result types
MappingMatch = Tuple[str, float, str]  # (label, score, metric)


@runtime_checkable
class FieldMappingResult(Protocol):
    best: MappingMatch
    candidates: List[MappingMatch]
