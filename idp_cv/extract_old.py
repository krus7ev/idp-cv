import logging
import math
import re
from collections import defaultdict
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import spacy
import torch
from dateutil import parser as date_parser
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .schemas import InvoiceSummary, TableLineItem
from .types import (
    DoclingBBox,
    DoclingDoc,
    DoclingGroup,
    DoclingProv,
    DoclingText,
    DoclingTextRef,
    FieldMappingResult,
    Ngram,
    TextID,
)
from .utils import (
    LexicalMapper,
    SemanticMapper,
    check_gpu_compatibility,
    clean_string,
    map_labels_to_fields,
)

logger = logging.getLogger(__name__)


class DocumentFieldExtractor:
    """
    Extracts key-value and floating fields in docling-converted documents (outside tables)
    Based on pydantic Field schema for key-mapping, value-parsing and post-processing
    """

    def __init__(
        self,
        fields: Sequence[FieldInfo],
        smapper: SemanticMapper,
        lmapper: LexicalMapper,
        ner: spacy.Language,
        device: str,
        threshold: float,
    ):
        self.fields = fields
        self.smapper = smapper
        self.lmapper = lmapper
        self.ner = ner
        self.device = device
        self.threshold = threshold

        self.field_value_types: Dict[str, str] = {}
        self._set_summary_field_value_types()
        self.text_data_buffer: Dict[int, str] = {}

    @staticmethod
    def _get_bbox(t: Union[DoclingText, DoclingGroup]) -> Optional[DoclingBBox]:
        # Retrieve bbox from prov. usually prov[0].bbox
        provs: List[DoclingProv] = getattr(t, 'prov', [])
        if provs and hasattr(provs[0], 'bbox'):
            return provs[0].bbox
        return None

    @staticmethod
    def _get_bbox_center(bbox: DoclingBBox) -> Tuple[float, float]:
        """Get center (x, y) of a bounding box."""
        # docling bbox has l, r, t, b attributes
        return ((bbox.l + bbox.r) / 2, (bbox.t + bbox.b) / 2)

    @classmethod
    def _euclidean_dist(cls, bbox1: DoclingBBox, bbox2: DoclingBBox) -> float:
        """Calculate Euclidean distance between centers of two bboxes."""
        c1 = cls._get_bbox_center(bbox1)
        c2 = cls._get_bbox_center(bbox2)
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    @classmethod
    def _is_horizontally_aligned(cls, bbox1: DoclingBBox, bbox2: DoclingBBox, tolerance: float = 10.0) -> bool:
        """
        Check if two bboxes are roughly on the same horizontal line (center y)
        tolerance can be dynamic based on height, but fixed is simpler
        """
        c1 = cls._get_bbox_center(bbox1)
        c2 = cls._get_bbox_center(bbox2)
        return abs(c1[1] - c2[1]) < tolerance

    @classmethod
    def _is_vertically_aligned(cls, bbox1: DoclingBBox, bbox2: DoclingBBox, tolerance: float = 50.0) -> bool:
        """Check if two bboxes are roughly in the same vertical column (center x)."""
        c1 = cls._get_bbox_center(bbox1)
        c2 = cls._get_bbox_center(bbox2)
        return abs(c1[0] - c2[0]) < tolerance

    @staticmethod
    def _resolve_text(
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

    @staticmethod
    def _process_candidates_ngrams(texts: List[DoclingText]) -> List[Tuple[TextID, List[Ngram]]]:
        """Prepare (text_id, n-grams) tuples for key candidate generation from a group of docling text objects"""
        key_cands: List[Tuple[TextID, List[Ngram]]] = []
        for text_obj in texts:
            # NOTE might afect matching back ngram in original text in map_field_keys! (corner case)
            tokens = [clean_string(t) for t in clean_string(text_obj.text).split(' ') if clean_string(t)]

            # Produce uni-gram, bi-gram and tri-gram candidates from text region
            for t, token in enumerate(tokens):
                colon_split = token.split(':', maxsplit=1)
                if len(colon_split) == 2 and colon_split[1]:
                    tokens = tokens[:t] + [colon_split[0] + ':', colon_split[1]] + tokens[t + 1 :]
            ngrams: List[Ngram] = []
            for n in range(1, min(len(tokens), 3) + 1):
                ngrams.extend([Ngram(' '.join(tokens[i : i + n])) for i in range(len(tokens) - n + 1)])

            text_id = TextID(int(text_obj.self_ref.split('/')[-1]))
            key_cands.append((text_id, ngrams))

        return key_cands

    @staticmethod
    def _process_text_start_ngrams(text: str, n_max: int = 3, include_full: bool = False) -> List[str]:
        """Generate n-grams starting from the first token of the text."""
        tokens = [clean_string(t) for t in text.split() if clean_string(t)]
        ngrams = []
        for n in range(1, min(len(tokens), n_max) + 1):
            ngrams.append(' '.join(tokens[:n]))
        if include_full and text not in ngrams:
            ngrams.append(text)
        return ngrams

    def _set_summary_field_value_types(self):
        for field in self.fields:
            if field.json_schema_extra:
                self.field_value_types[field.alias] = field.json_schema_extra.get('value_type', 'string')

    def _log_key_mapping_debug_results(
        self,
        ngram_match_by_field: Dict[str, FieldMappingResult],
        ngrams_map: Dict[str, List[Tuple[int, int]]],
        threshold: float,
    ) -> None:
        """Helper to log detailed mapping information for all fields."""
        logger.debug('Field Key Mapping Results:')
        for field_name, match in ngram_match_by_field.items():
            best_ngram, score, metric = match['best']
            if score < threshold:
                continue

            locations = ngrams_map[best_ngram]
            loc_details = [f'group[{locations[i][0]}]' for i in range(min(len(locations), 3))]
            if len(locations) > 3:
                loc_details.append('...')

            logger.debug(
                f'\n  - Field "{field_name}" matched key "{best_ngram}" (score: {score:.2f} [{metric}])'
                f' at {len(locations)} locations: {", ".join(loc_details)}'
            )

            if len(match['candidates']) > 1:
                alt_cands = [f"'{c[0]}' ({c[1]:.2f} [{c[2]}])" for c in match['candidates'][:5]]
                logger.debug(f'    - Candidates considered:\n    {", ".join(alt_cands)}')

        # Log unmatched fields that were below threshold but were "candidates"
        field_names = list([f.alias for f in self.fields])
        unmatched = [
            f for f in field_names if f not in ngram_match_by_field or ngram_match_by_field[f]['best'][1] < threshold
        ]
        if unmatched:
            logger.debug(f'\n  => Unmatched Fields: {", ".join(unmatched)}\n')

    def _initialize_text_buffer(
        self, doc: DoclingDoc, fields_by_ngram: Dict[Ngram, Tuple[str, List[Tuple[int, TextID]]]]
    ) -> None:
        """Initializes and populates the text buffer, removing identified field keys."""
        texts = getattr(doc, 'texts', [])
        self.text_data_buffer = {i: texts[i].text for i in range(len(texts))}
        for field_ngram, (_, locations) in fields_by_ngram.items():
            for _, tid in locations:
                # Ensure "Inv No:" matches even if field_ngram is "inv no"
                pattern = re.escape(field_ngram).replace(r'\ ', r'\s*')
                text_minus_data = re.sub(pattern, '', self.text_data_buffer[tid], flags=re.IGNORECASE).strip()
                if text_minus_data:
                    self.text_data_buffer[tid] = text_minus_data

    def _extract_and_consume_value(
        self, field_name: str, cand_tid: TextID, cand_text: str, strategy: str
    ) -> Optional[Union[str, float]]:
        """Common logic to generate n-grams, parse value, and consume from buffer."""
        cand_start_ngrams = self._process_text_start_ngrams(cand_text, n_max=3, include_full=True)
        logger.debug(f'    - Strategy {strategy} ngram candidates: {cand_start_ngrams}')
        for ngram in reversed(cand_start_ngrams):
            parsed = self.parse_value(field_name, ngram)
            if parsed is not None:
                logger.debug(f"      [MATCH] Strategy {strategy}: Found value '{parsed}' from '{ngram}'")

                pattern = re.escape(ngram).replace(r'\ ', r'\s*')
                self.text_data_buffer[cand_tid] = re.sub(
                    pattern, '', self.text_data_buffer[cand_tid], count=1, flags=re.IGNORECASE
                ).strip()
                if not self.text_data_buffer[cand_tid]:
                    del self.text_data_buffer[cand_tid]

                return parsed
        return None

    def _extract_value_from_text(
        self, field_name: str, field_ngram: str, cand_tid: TextID, original_text: str
    ) -> Optional[Union[str, float]]:
        """Strategy A: In-line mapping (Value is in the same text block as the Key)."""
        if cand_tid not in self.text_data_buffer or not self.text_data_buffer[cand_tid].strip():
            return None

        parts = re.split(re.escape(field_ngram), original_text, flags=re.IGNORECASE)
        cand_text = clean_string(parts[-1]) if len(parts) > 1 else ''

        if cand_text:
            return self._extract_and_consume_value(field_name, cand_tid, cand_text, 'A')

        return None

    def _extract_value_from_group(
        self,
        field_name: str,
        tid: TextID,
        g: int,
        texts: List[DoclingText],
        candidates: List[List[Tuple[TextID, List[Ngram]]]],
    ) -> Optional[Union[str, float]]:
        """Strategy B: Spatial mapping (Value is in a separate text block)."""
        key_bbox = self._get_bbox(texts[tid])
        if not key_bbox:
            return None

        group_cands = []
        # Search within the SAME group defined by Docling
        for neighbor_tid, _ in candidates[g]:
            if neighbor_tid == tid or neighbor_tid not in self.text_data_buffer:
                continue

            neighbor_text = texts[neighbor_tid]
            neighbor_bbox = self._get_bbox(neighbor_text)
            if not neighbor_bbox:
                continue

            is_horz = self._is_horizontally_aligned(key_bbox, neighbor_bbox)
            is_vert = self._is_vertically_aligned(key_bbox, neighbor_bbox)

            # To the RIGHT or BELOW
            is_right = is_horz and neighbor_bbox.l >= key_bbox.l
            is_below = is_vert and neighbor_bbox.t <= key_bbox.t

            if is_right or is_below:
                dist = self._euclidean_dist(key_bbox, neighbor_bbox)
                # Weight right-hand matches more heavily than top-down
                score = dist * (1.0 if is_right else 1.5)
                group_cands.append((self.text_data_buffer[neighbor_tid], score, neighbor_tid))

        # Sort by proximity
        group_cands.sort(key=lambda x: x[1])
        logger.debug(
            f'    - Strategy B shortlist {len(group_cands)} candidates of group #{g}: {[c[0] for c in group_cands]}'
        )
        for cand_text, _, cand_tid in group_cands:
            parsed = self._extract_and_consume_value(field_name, cand_tid, cand_text, 'B')
            if parsed is not None:
                return parsed

        return None

    def _extract_orphan_value(self, field_name: str) -> Optional[Union[str, float]]:
        """Strategy C: Orphan Remainder Recovery from unused text buffer."""
        for tid, remaining_text in list(self.text_data_buffer.items()):
            if not remaining_text:
                continue

            parsed = self._extract_and_consume_value(field_name, tid, remaining_text, 'C')
            if parsed:
                return parsed
        return None

    def extract_candidates(self, doc: DoclingDoc) -> List[List[Tuple[TextID, List[Ngram]]]]:
        """Extract key candidates from doc groups and orphan texts."""
        candidates: List[List[Tuple[TextID, List[Ngram]]]] = []

        groups = doc.groups
        texts = doc.texts

        # NOTE: We consider all groups even if their label isn't 'key_value_area' here
        processed_texts: Set[int] = set()
        for group in groups:
            group_texts = []
            children = group.children
            for child in children:
                t_obj = self._resolve_text(child, doc, texts)
                if t_obj:
                    group_texts.append(t_obj)
                    processed_texts.add(id(t_obj))

            if group_texts:
                candidates.append(self._process_candidates_ngrams(group_texts))

        # NOTE: We consider all other orphans texts here - no check if "text.parent" is "body"
        orphan_texts = []
        for text in texts:
            if id(text) not in processed_texts:
                orphan_texts.append(text)

        # Treat orphans as an extra group
        if orphan_texts:
            candidates.append(self._process_candidates_ngrams(orphan_texts))

        return candidates

    def map_field_keys(
        self,
        candidates: List[List[Tuple[TextID, List[Ngram]]]],
        threshold: float = 0.60,
        lex_threshold: float = 0.85,
    ) -> Dict[Ngram, Tuple[str, List[Tuple[int, TextID]]]]:

        # Map each n-gram to a list of its locations to prevent overwriting
        ngrams_map: Dict[Ngram, List[Tuple[int, TextID]]] = defaultdict(list)
        for g, candidates_group in enumerate(candidates):
            for text_id, text_ngrams in candidates_group:
                for ngram in text_ngrams:
                    ngrams_map[ngram].append((g, text_id))

        ngram_match_by_field = map_labels_to_fields(
            labels=ngrams_map.keys(),
            smapper=self.smapper,
            lmapper=self.lmapper,
            lexical_threshold=lex_threshold,
        )

        if logger.isEnabledFor(logging.DEBUG):
            self._log_key_mapping_debug_results(ngram_match_by_field, ngrams_map, threshold)

        field_and_origin_by_ngram = {
            Ngram(res['best'][0]): (field_name, ngrams_map[Ngram(res['best'][0])])
            for field_name, res in ngram_match_by_field.items()
            if res['best'][1] >= threshold
        }
        return field_and_origin_by_ngram

    def map_field_values(
        self,
        doc: DoclingDoc,
        fields_by_ngram: Dict[Ngram, Tuple[str, List[Tuple[int, TextID]]]],
        candidates: List[List[Tuple[TextID, List[Ngram]]]],
    ) -> Dict[str, Union[str, float, None]]:
        """
        Resolves keys to values using a three-tier extraction strategy:

        1. Strategy A (In-line): Matches values within the same text block as the key
        2. Strategy B (Spatial): Searches neighboring blocks in the same group using
           proximity and alignment (horizontal/vertical) heuristics
        3. Strategy C (Orphan): Recovers missing 'name' or 'address' fields from
           remaining unconsumed blocks without explicit keys

        Args:
            doc: Document object with text elements and bboxes
            fields_by_ngram: Map of key n-grams to (field_name, [(group_id, text_id)])
            candidates: Nested list of (text_id, n-grams) grouped by spatial relevance

        Returns:
            Dict mapping field names to parsed values
        """
        texts = getattr(doc, 'texts', [])
        summary_data: Dict[str, Union[str, float, None]] = {}

        # Text buffer to keep track of remaining text after key/value substring consumption
        self._initialize_text_buffer(doc, fields_by_ngram)

        # Match values for each identified field
        logger.debug('Resolving field values (Strategy A: same-text RHS; B: Neighbour texts in group):')
        for field_ngram, (field_name, locations) in fields_by_ngram.items():
            logger.debug(f"Field '{field_name}' (Key: '{field_ngram}')")

            extra_field = None
            if field_name.endswith('tax'):
                extra_field = 'receiver_tax' if field_name.startswith('issuer') else 'issuer_tax'
            elif field_name.endswith('addr'):
                extra_field = 'receiver_addr' if field_name.startswith('issuer') else 'issuer_addr'

            value_match: Union[str, float, None] = None
            values_matched = []
            for g, tid in locations:
                # Strategy A: In-line mapping
                value_match = self._extract_value_from_text(field_name, field_ngram, tid, texts[tid].text)

                # Strategy B: Spatial mapping
                if value_match is None:
                    value_match = self._extract_value_from_group(field_name, tid, g, texts, candidates)

                # If field value is found near current key location, stop checking the key in other locations
                if value_match is not None:
                    values_matched.append((value_match, field_name))
                    if extra_field is None:
                        break
                    else:
                        logger.debug(f"  Extra Field '{extra_field}' (Key: '{field_ngram}')")
                        field_name = extra_field
                        extra_field = None

            if values_matched:
                for value, field in values_matched:
                    summary_data[field] = value
            else:
                logger.debug(f'    - No value found across {len(locations)} key locations')

        # Extract values whithout an nearby key from unused text buffer if they fit any missing fields
        missing_fields = [f for f in self.field_value_types if f not in summary_data]
        if missing_fields:
            logger.debug(f'Strategy C: Attempting orphan-value recovery for missing fields: {missing_fields}')
            for field_name in missing_fields:
                parsed = self._extract_orphan_value(field_name)
                if parsed:
                    summary_data[field_name] = parsed

        return summary_data

    def parse_value(self, field_name: str, value_text: str) -> Union[str, float, None]:
        """
        Parses a candidate value string based on the expected value type for the field,
        using parsing heuristics and NER validation.
        """
        v_type = self.field_value_types.get(field_name, 'string')

        clean_text = clean_string(value_text)  # strip=": ")
        if not clean_text or len(clean_text) < 2:
            return None

        # Label Rejection Collision Check: Do not pick up labels as values!
        # This addresses the Strategy B label-capture in inv3 and inv4
        lex_match, lex_score = self.lmapper.map_field(clean_text)
        if lex_match and lex_score > 0.8:
            logger.debug(f"      Rejected value candidate '{clean_text}' - matches '{lex_match}' field by {lex_score}")
            return None

        for label in self.lmapper.all_aliases:
            if label.lower() in clean_text.lower().split():
                logger.debug(f"      Rejected value candidate '{clean_text}' - contains alias '{label}'")
                return None

        is_pure_numeric = bool(re.fullmatch(r'[\d\s\.,\$\€\%\-\(\)]+', clean_text))

        value = None
        num_parse = TableLineItem.parse_numeric(clean_text)
        if v_type in ('amount', 'rate') and is_pure_numeric:
            value = num_parse
        elif num_parse and is_pure_numeric:
            value = None

        if v_type == 'id':
            # Validate as an identifier: Alphanumeric with common delimiters
            # Based on examples: "INV-2024-001", "5414", "TX-987654"
            if re.fullmatch(r'[A-Z0-9\-_./#\s]{2,40}', clean_text, re.IGNORECASE):
                # Ensure it's not JUST spaces or punctuation
                if any(c.isalnum() for c in clean_text):
                    value = clean_text

        elif v_type == 'date' and num_parse:
            if int(re.sub(r'\D', '', clean_text)) < 32:
                return None
            try:
                date = date_parser.parse(clean_text, fuzzy=False)
                value = date.strftime('%Y-%m-%d')
            except (ValueError, IndexError, OverflowError):
                pass

        elif v_type in ('name', 'address') and not is_pure_numeric:
            # 1. Run NER on the candidate
            doc = self.ner(clean_text)
            ents = [ent.label_ for ent in doc.ents]

            # 2. Validation Logic
            is_valid = False
            if v_type == 'name':
                # Names: Must have PERSON, ORG, or NORP
                # CRITICAL: If it has GPE or FAC, it's likely an address or includes one, so reject for 'name'
                if all(ent in {'ORG', 'PERSON', 'NORP'} for ent in ents):
                    if not any(ext in ents for ext in {'GPE', 'FAC', 'LOC'}):
                        is_valid = True

            elif v_type == 'address':
                # Addresses: Must have GPE, LOC, or FAC
                # It's okay if an address contains an ORG (e.g. "Acme Corp, 123 Street")
                if all(ent in {'GPE', 'LOC', 'FAC', 'CARDINAL', 'ORG'} for ent in ents):
                    if not any(ent in {'PERSON', 'NORP'} for ent in ents):
                        is_valid = True
                # # Heuristic fallback for addresses that NER misses but look like street lines
                # elif len(ents) > 2 and all(
                #   ent in {'GPE', 'LOC', 'FAC', 'CARDINAL', 'ORG'} for ent in [ents[0], ents[-1]]
                # ):
                #      if not any ()
                #      is_valid = True

            if is_valid and re.fullmatch(r'[A-Z0-9\s.,\-#\'/&()]+', clean_text, re.IGNORECASE):
                # Ensure it's not just a number or random noise - names/addresses have letters
                if any(c.isalpha() for c in clean_text):
                    value = clean_text
                    if v_type == 'name':
                        text_tokens = clean_text.split()
                        if len(text_tokens) > 2 and not any(
                            text_tokens[-1].lower().startswith(suffix)
                            for suffix in {'llc', 'ltd', 'inc', 'gmbh', 'corp'}
                        ):
                            value = None
                            logger.debug(
                                f"      Rejected value candidate '{clean_text}' - Too long for non-INC."
                                f" - for field of value-type '{v_type}'"
                            )
                else:
                    logger.debug(
                        f"      Rejected value candidate '{clean_text}' - No alpha chars"
                        f" - for field of value-type '{v_type}'"
                    )
            else:
                logger.debug(
                    f"      Rejected value candidate '{clean_text}' - Fails Regex - for field of value-type '{v_type}'"
                )

        if value is None:
            logger.debug(f"      Rejected '{clean_text}' - failed conversion for value-type '{v_type}'")
            pass

        return value

    @classmethod
    def create(
        cls,
        schema: Union[Sequence[FieldInfo], Type[BaseModel]] = InvoiceSummary,
        smapper=None,
        lmapper=None,
        ner=None,
        device='cpu',
        threshold=0.60,
    ) -> Optional['DocumentFieldExtractor']:
        if not schema or (
            not (isinstance(schema, type) and issubclass(schema, BaseModel))
            and not (isinstance(schema, Sequence) and isinstance(schema[0], FieldInfo))
        ):
            return
        fields = schema if isinstance(schema, Sequence) else list(schema.model_fields.values())

        if device != 'cpu':
            device = torch.device(check_gpu_compatibility(device))

        if smapper is None:
            smapper = SemanticMapper.create(fields, device=device)
        if lmapper is None:
            lmapper = LexicalMapper.create(fields)
        if ner is None:
            ner = spacy.load('en_core_web_md')

        return cls(fields, smapper, lmapper, ner, device, threshold)


def extract_invoice_summary(
    doc: DoclingDoc,
    # line_items: Optional[List[List[TableLineItem]]] = None
    extractor: Optional[DocumentFieldExtractor] = None,
    smapper: Optional[SemanticMapper] = None,
    lmapper: Optional[LexicalMapper] = None,
    device: str = 'cpu',
    threshold: float = 0.60,
) -> InvoiceSummary:
    """Extract invoice summary fields from a document using semantic + rule-based scoring."""
    if extractor is None:
        extractor = DocumentFieldExtractor.create(InvoiceSummary, smapper, lmapper, device, threshold)

    if extractor is None:
        raise ValueError('Failed to create extractor')

    candidates = extractor.extract_candidates(doc)
    fields_by_key = extractor.map_field_keys(candidates)

    mapped_data = extractor.map_field_values(doc, fields_by_key, candidates)

    # We construct the model, pydantic will handle extra validation or defaults
    return InvoiceSummary.model_validate(mapped_data)
