import logging
import math
import os
import re
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import pandas as pd
import torch
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer

from .constants import DEFAULT_GRANITE_MODEL_ID
from .types import (
    DoclingBBox,
    DoclingDoc,
    DoclingEntity,
    DoclingGroup,
    DoclingPage,
    DoclingPageSize,
    DoclingProv,
    DoclingText,
    DoclingTextRef,
)

logger = logging.getLogger(__name__)


def load_embedding_model(model_id: str = DEFAULT_GRANITE_MODEL_ID, device='cpu') -> Tuple[SentenceTransformer, str]:
    """Load embedding model with offline-aware fallback to local cache."""

    if device != 'cpu':
        device = check_gpu_compatibility(device)

    local_files_only = os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'
    try:
        model = SentenceTransformer(model_id, device=device, local_files_only=local_files_only)
    except Exception as exc:
        if not local_files_only:
            logger.warning(
                f"Could not load model '{model_id}' from HuggingFace (Network error?). Trying local cache..."
            )
            model = SentenceTransformer(model_id, device=device, local_files_only=True)
        else:
            raise exc

    return model, device


def check_gpu_compatibility(device: str, min_compute_capability: tuple = (7, 0)) -> str:
    """
    Check if GPU supports required compute capability for torch.compile
    Falls back to CPU for incompatible GPUs

    Args:
        device: Requested device ('cpu', 'cuda', 'cuda:0', etc.)
        min_compute_capability: Minimum required (major, minor) version

    Returns:
        Device string to use ('cpu', 'cuda', 'cuda:0', etc.)
    """
    if not device.startswith('cuda'):
        return device

    if not torch.cuda.is_available():
        return 'cpu'

    gpu_id = 0 if device == 'cuda' else int(device.split(':')[-1])
    cc_major, cc_minor = torch.cuda.get_device_capability(gpu_id)
    min_major, min_minor = min_compute_capability

    if cc_major < min_major or (cc_major == min_major and cc_minor < min_minor):
        gpu_name = torch.cuda.get_device_name(gpu_id)
        logger.warning(f'[{gpu_name}] (CC {cc_major}.{cc_minor}) < {min_compute_capability} → defaulting to CPU')
        return 'cpu'

    return device


def draw_pdf_bbox_on_image(
    draw: ImageDraw.Draw,
    bbox: DoclingBBox,
    pdf_size: DoclingPageSize,
    image_shape: tuple,
    color: str,
    thickness: int,
    flip: bool = True,
):
    """
    Draws a bounding box in PDF coordinates onto an image using PIL's ImageDraw
        - pdf_size: page_item.size (from JSON)
        - image_shape: img.size (PIL pixels)
    """
    scale_x = image_shape[0] / pdf_size.width
    scale_y = image_shape[1] / pdf_size.height

    # Scale and flip (PDF is bottom-up, PIL is top-down)
    left: int = bbox.l * scale_x
    right: int = bbox.r * scale_x
    top = (pdf_size.height - bbox.t) if flip is True else bbox.t
    bottom = (pdf_size.height - bbox.b) if flip is True else bbox.b
    top *= scale_y
    bottom *= scale_y

    # Enforce increasing order for PIL compliance (sanity)
    x0, x1 = min(left, right), max(left, right)
    y0, y1 = min(top, bottom), max(top, bottom)

    draw.rectangle([x0, y0, x1, y1], outline=color, width=thickness)


def draw_page_items(
    image: Image.Image,
    entities: List[DoclingEntity],
    page_no: int,
    page_item: DoclingPage,
    color: str = 'red',
    thickness: int = 1,
    flip: bool = True,
):
    """Draws bboxes for any entity list that exposes a `prov` (list[ProvenanceItem]) attribute."""
    draw = ImageDraw.Draw(image)
    for ent in entities:
        for prov in getattr(ent, 'prov', []):
            if prov.page_no == page_no:
                draw_pdf_bbox_on_image(draw, prov.bbox, page_item.size, image.size, color, thickness, flip)


def clean_string(value: object, *, lowercase: bool = False, strip=' ', splitup=False) -> str:
    """Standard text cleaner used across extraction and matching flows."""
    if not value:
        return ''

    text = str(value).replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip(strip)

    if splitup is True:
        # Add space around punctuation (like colons/commas) if missing
        # Captures letter/digit before punctuation, and letter/digit after
        text = re.sub(r'([A-Za-z0-9])([:,;])([A-Za-z0-9])', r'\1\2 \3', text)

        # Split between lowercase and uppercase (CamelCase splitting)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # Split between digits and letters
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)

    return text.lower() if lowercase else text


def normalize_text(value: str) -> str:
    """Normalize text for lexical matching comparisons."""
    return clean_string(value, lowercase=True)


def extract_clean_table_data(
    doc: DoclingDoc,
    keywords: set = {'item', 'description', 'desc', 'qty', 'quantity', 'rate', 'unit', 'price', 'amount', 'total'},
) -> List[List[Dict]]:
    """
    Extracts invoice-like tables and keeps each table as a separate list of rows
    Returns: [table_1_rows, table_2_rows, ...]
    """
    clean_tables: List[List[Dict]] = []

    for table_item in doc.tables:
        df = table_item.export_to_dataframe()
        if len(df.columns) < 2 or len(df) < 2:
            continue

        # Normalize column names with same logic as lexical matching
        df.columns = [normalize_text(c) for c in df.columns]
        # if not any(any(key in col for key in keywords) for col in df.columns):
        #     continue

        table_rows: List[Dict] = []
        for row in df.to_dict(orient='records'):
            clean_row = {col: (normalize_text(row.get(col)) if pd.notna(row.get(col)) else '') for col in df.columns}
            table_rows.append(clean_row)

        if table_rows:
            clean_tables.append(table_rows)

    extra_tables = []
    for table in clean_tables:
        vertical_tab = {}
        for row in reversed(table):
            row_values = list(row.values())
            if row_values[-1] and '' in row_values:
                extra_data = [data for data in row_values[:-1] if data]
                if len(extra_data) == 1:
                    vertical_tab.update({extra_data[0]: row_values[-1]})
        if vertical_tab:
            extra_tables.append([vertical_tab])

    if extra_tables:
        clean_tables.extend(extra_tables)

    return clean_tables


def get_bbox(t: Union[DoclingText, DoclingGroup]) -> Optional[DoclingBBox]:
    """Retrieve bbox from prov. (expected in prov[0].bbox)"""
    provs: List[DoclingProv] = getattr(t, 'prov', [])
    if provs and hasattr(provs[0], 'bbox'):
        return provs[0].bbox
    return None


def get_bbox_center(bbox: DoclingBBox) -> Tuple[float, float]:
    """
    Get center (x, y) of a Docling bounding box.
        bbox: has l, r, t, b coordinate attributes in BOTTOMLEFT origin format
    """
    return ((bbox.l + bbox.r) / 2, (bbox.t + bbox.b) / 2)


def euclidean_dist(bbox1: DoclingBBox, bbox2: DoclingBBox) -> float:
    """Calculate Euclidean distance between centers of two bboxes."""
    c1, c2 = get_bbox_center(bbox1), get_bbox_center(bbox2)
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def vertically_overlap(bbox1: DoclingBBox, bbox2: DoclingBBox) -> bool:
    "Check if two bounding boxes' vertical spans overlap"
    return max(bbox1.b, bbox2.b) < min(bbox1.t, bbox2.t)


def horizontally_overlap(bbox1: DoclingBBox, bbox2: DoclingBBox) -> bool:
    "Check if two bounding boxes' horizontal spans overlap"
    return min(bbox1.r, bbox2.r) > max(bbox1.l, bbox2.l)


def is_horizontally_aligned(bbox1: DoclingBBox, bbox2: DoclingBBox) -> bool:
    """
    Check if two bboxes are next to each other with no overlapping area (disjoint)"
    """
    return vertically_overlap(bbox1, bbox2) and not horizontally_overlap(bbox1, bbox2)


def is_vertically_aligned(bbox1: DoclingBBox, bbox2: DoclingBBox) -> bool:
    "Check if two bounding boxes are one above the other with no overlapping area (disjoint)"
    return horizontally_overlap(bbox1, bbox2) and not vertically_overlap(bbox1, bbox2)


def resolve_text(
    ref: Union[DoclingTextRef, Dict[str, str]], doc: DoclingDoc, texts: List[DoclingText]
) -> Optional[DoclingText]:
    """Helper to retrieve text object from document"""
    obj: Optional[DoclingText] = None
    if hasattr(ref, 'resolve'):
        obj = ref.resolve(doc)
    elif isinstance(ref, dict) and '$ref' in ref:
        # JSON Pointer resolution
        parts = ref['$ref'].split('/')
        if len(parts) > 2 and parts[1] == 'texts':
            try:
                idx = int(parts[2])
                if 0 <= idx < len(texts):
                    obj = texts[idx]
            except (ValueError, IndexError):
                pass

    return obj if hasattr(obj, 'text') else None
