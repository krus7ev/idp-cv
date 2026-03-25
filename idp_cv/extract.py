import logging
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
from dateutil import parser as date_parser
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .constants import (
    ADDR_NER_TAGS,
    C_SWAP_FILEDS,
    COMPANY_SUFFIXES,
    LOC_NER_TAGS,
    NAME_NER_TAGS,
    RE_ADDR_ALPHANUMERIC,
    RE_DOUBLE_DIGIT_PERCENTAGE,
    RE_ID_ALPHANUMERIC,
    RE_IS_PURE_NUMERIC,
)
from .parse import (
    LexicalMapper,
    SemanticMapper,
    map_labels_to_fields,
)
from .schemas import InvoiceSummary, TableLine
from .types import (
    DoclingDoc,
    DoclingText,
    FieldMappingResult,
)
from .utils import (
    check_gpu_compatibility,
    clean_string,
    euclidean_dist,
    get_bbox,
    is_horizontally_aligned,
    is_vertically_aligned,
    resolve_text,
)

logger = logging.getLogger(__name__)


class DocumentFieldExtractor:
    """
    Extracts key-value and floating fields in docling-converted documents (outside tables)
    Based on pydantic Field schema for key-mapping, value-parsing and post-processing
    """

    # Class-level caches to share data across all documents and instances
    _ner_cache: Dict[str, List[str]] = {}
    _parse_cache: Dict[Tuple[str, str], Optional[Union[str, float]]] = {}

    def __init__(
        self,
        fields: Sequence[FieldInfo],
        smapper: SemanticMapper,
        lmapper: LexicalMapper,
        ner: spacy.Language,
        threshold: float,
    ):
        self.fields = fields
        self.smapper = smapper
        self.lmapper = lmapper
        self.ner = ner
        self.threshold = threshold

        self.field_value_types: Dict[str, str] = {}
        self._set_summary_field_value_types()
        self.text_data_buffer: Dict[int, str] = {}

    @staticmethod
    def _process_text_ngrams(
        text: str, n_max: int = 3, include_full: bool = False, start=False, reverse: bool = False
    ) -> List[str]:
        """Generate n-grams starting from the first token of the text."""
        # Clean naively (without extra splitup=True parameter), taking care only of colons
        clean_text = clean_string(text)
        space_split = clean_text.split()
        tokens = []
        for token in space_split:
            colon_split = token.split(':')
            if len(colon_split) >= 2:
                tokens.extend([ct + ':' for ct in colon_split[:-1] if ct])
                if colon_split[-1]:
                    tokens.append(colon_split[-1])
            else:
                tokens.append(token)

        clean_text = ' '.join(tokens)

        ngrams: List[str] = []
        for n in range(1, min(len(tokens), n_max) + 1):
            if start:
                ngrams.append(' '.join(tokens[:n]))
            else:
                ngrams.extend([' '.join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])

        if include_full and clean_text not in ngrams:
            ngrams.append(clean_text)

        if reverse is True:
            ngrams.reverse()

        return ngrams

    @classmethod
    def _process_key_candidate_ngrams(cls, texts: List[DoclingText], n_max: int = 3) -> List[Tuple[int, List[str]]]:
        """Prepare (text_id, n-grams) tuples for key candidate generation from a group of docling text objects"""
        key_cands: List[Tuple[int, List[str]]] = []
        for text_obj in texts:
            ngrams = cls._process_text_ngrams(text_obj.text, n_max=n_max)
            text_id = int(text_obj.self_ref.split('/')[-1])
            key_cands.append((text_id, ngrams))

        return key_cands

    @staticmethod
    def _swap_field_pairs(field_pairs: List[Tuple[str]], fields: List[str]) -> List[str]:
        for f1, f2 in field_pairs:
            if f1 in fields and f2 in fields:
                logger.debug(f"  Swapping order of '{f1}' and '{f2}' in missing fields")
                f1i, f2i = fields.index(f1), fields.index(f2)
                fields[f1i], fields[f2i] = fields[f2i], fields[f1i]

        return fields

    @staticmethod
    def _extract_tax_rate_from_amount_key(key_ngram: str) -> Optional[float]:
        rate = re.search(RE_DOUBLE_DIGIT_PERCENTAGE, key_ngram)
        if rate:
            return float(rate.group(0).strip('%')) / 100.0

    @classmethod
    def _process_tax_field(
        cls, field: str, ngram: str, match: str, matches_map: Dict[str, Union[str, float, None]]
    ) -> str:
        """Tax field post-processing heuristic: adds 'tax_rate' to matches list if found in key; returns final match"""
        if field == 'tax_amount':
            tax_rate = cls._extract_tax_rate_from_amount_key(ngram)
            if tax_rate is not None:
                matches_map['tax_rate'] = tax_rate
        elif field == 'tax_rate' and match > 1.0:
            match /= 100.0
        return match

    @staticmethod
    def _get_bounded_regex_pattern(ngram: str) -> str:
        """Constructs a regex pattern for an ngram with conditional word boundaries."""
        escaped = re.escape(ngram).replace(r'\ ', r'\s*')
        prefix = r'\b' if ngram and re.match(r'\w', ngram) else ''
        suffix = r'\b' if ngram and re.search(r'\w$', ngram) else ''
        return f'{prefix}{escaped}{suffix}'

    def _set_summary_field_value_types(self):
        for field in self.fields:
            if field.json_schema_extra:
                self.field_value_types[field.alias] = field.json_schema_extra.get('value_type', 'string')

    def _initialize_text_buffer(
        self, doc: DoclingDoc, fields_map: Dict[str, Tuple[str, List[Tuple[int, int]]]]
    ) -> None:
        """Initializes and populates the text buffer, removing identified field keys."""
        texts = getattr(doc, 'texts', [])
        self.text_data_buffer = {i: texts[i].text for i in range(len(texts))}
        # for field_ngram, (_, locations) in fields_by_ngram.items():
        for field_ngram, locations in fields_map.values():
            for _, tid in locations:
                if tid not in self.text_data_buffer:
                    continue

                pattern = self._get_bounded_regex_pattern(field_ngram)

                text_minus_data = re.sub(pattern, '', self.text_data_buffer[tid], flags=re.IGNORECASE).strip()
                if text_minus_data:
                    self.text_data_buffer[tid] = text_minus_data
                else:
                    del self.text_data_buffer[tid]

    def _parse_value(self, field_name: str, value_text: str) -> Optional[Union[str, float]]:
        """
        Retrieves the memoized parsed value, or computes it using parsing heuristics.
        """
        clean_text = clean_string(value_text)
        if not clean_text or len(clean_text) < 2:
            return

        v_type = self.field_value_types.get(field_name, 'string')
        cache_key = (v_type, clean_text)
        if cache_key in self._parse_cache:
            return self._parse_cache[cache_key]

        result = self._parse_value_heuristics(v_type, clean_text)
        self._parse_cache[cache_key] = result
        return result

    def _parse_value_heuristics(self, v_type: str, clean_text: str) -> Optional[Union[str, float]]:
        """
        Parses a candidate value string based on the expected value type for the field,
        using evaluation heuristics and NER validation.
        """
        # 1. Label rejection: Discard value candidates containing known field labels
        for label in self.lmapper.all_aliases:
            if label.lower().strip(' :') in clean_text.lower().split():
                logger.debug(f"      Rejected value candidate '{clean_text}' - contains alias '{label}'")
                return

        # 2. Label rejection: Do not pick up labels as values
        lex_match, lex_score = self.lmapper.map_field(clean_text)[0]
        if lex_match and lex_score > 0.8:
            logger.debug(f"      Rejected value candidate '{clean_text}' - matches '{lex_match}' field by {lex_score}")
            return

        # 3. Parse or reject numeric values based on expected value type
        has_digit = any(c.isdigit() for c in clean_text)
        is_pure_numeric = bool(RE_IS_PURE_NUMERIC.fullmatch(clean_text))
        num_parse = TableLine.parse_numeric(clean_text)
        text_tokens = clean_text.split()

        if v_type in ('amount', 'rate'):
            if is_pure_numeric:
                return num_parse
            return

        if v_type == 'id':
            # Validate as an identifier: Alphanumeric with common delimiters
            if re.fullmatch(RE_ID_ALPHANUMERIC, clean_text, re.IGNORECASE):
                # Ensure it's not JUST spaces or punctuation
                if has_digit and len(text_tokens) <= 2:
                    if len([c for c in clean_text if c.isdigit()]) > 3:
                        return clean_text

        elif v_type == 'date' and num_parse:
            if int(re.sub(r'\D', '', clean_text)) < 32:
                return
            try:
                date = date_parser.parse(clean_text, fuzzy=False)
                return date.strftime('%Y-%m-%d')
            except (ValueError, IndexError, OverflowError):
                return

        elif v_type in ('name', 'address') and not is_pure_numeric:
            # Run NER on the candidate
            if clean_text not in self._ner_cache:
                doc = self.ner(clean_text)
                self._ner_cache[clean_text] = [ent.label_ for ent in doc.ents]

            ents = self._ner_cache[clean_text]

            # Validation Logic
            if v_type == 'name':
                if all(ent in NAME_NER_TAGS for ent in ents):
                    if len(text_tokens) > 3 and not any(
                        text_tokens[-1].lower().startswith(suffix) for suffix in COMPANY_SUFFIXES
                    ):
                        logger.debug(
                            f"      Rejected value candidate '{clean_text}' - Too long to lack {COMPANY_SUFFIXES}"
                            f" - for field of value-type '{v_type}'"
                        )
                        return
                    elif has_digit:
                        logger.debug(
                            f"      Rejected value candidate '{clean_text}' - Contains digit(s)"
                            f" - for field of value-type '{v_type}'"
                        )
                        return

                    return clean_string(clean_text, splitup=True)
                else:
                    logger.debug(
                        f"      Rejected value candidate '{clean_text}' - Not all in {NAME_NER_TAGS}"
                        f" - for field of value-type '{v_type}: {ents}'"
                    )

            elif v_type == 'address':
                if all(ent in ADDR_NER_TAGS for ent in ents):
                    if re.fullmatch(RE_ADDR_ALPHANUMERIC, clean_text, re.IGNORECASE):
                        if (len(text_tokens) >= 2 and has_digit) or (
                            len(text_tokens) > 3 and all(ent in LOC_NER_TAGS for ent in ents)
                        ):
                            return clean_string(clean_text, splitup=True)
                        else:
                            logger.debug(
                                f"      Rejected value candidate '{clean_text}' - No digit for address"
                                f" and not enough entities in {LOC_NER_TAGS} - for value-type '{v_type}'"
                            )
                            return
                    else:
                        logger.debug(
                            f"      Rejected value candidate '{clean_text}' - Failed alphanumeric regex"
                            f" - for value-type '{v_type}'"
                        )
                else:
                    logger.debug(
                        f"      Rejected value candidate '{clean_text}' - Not all in {ADDR_NER_TAGS}"
                        f" - for value-type '{v_type}': {ents}"
                    )
            else:
                logger.debug(
                    f"      Rejected value candidate '{clean_text}' - Fails Regex - for field of value-type '{v_type}'"
                )

        return

    def _extract_and_consume_value(
        self, field_name: str, cand_ngrams: List[str], cand_tid: int, strategy: str
    ) -> Optional[Union[str, float]]:
        """Common logic to generate n-grams, parse value, and consume from buffer."""

        logger.debug(f'    - Strategy {strategy} ngram candidates: {cand_ngrams}')
        for ngram in cand_ngrams:
            parsed = self._parse_value(field_name, ngram)
            if parsed is not None:
                logger.debug(f"      [MATCH] Strategy {strategy}: Found value '{parsed}' from '{ngram}'")

                pattern = self._get_bounded_regex_pattern(ngram)

                self.text_data_buffer[cand_tid] = re.sub(
                    pattern, '', self.text_data_buffer[cand_tid], count=1, flags=re.IGNORECASE
                ).strip()
                if not self.text_data_buffer[cand_tid]:
                    del self.text_data_buffer[cand_tid]

                return parsed
        return None

    def _extract_value_from_text(
        self, field_name: str, field_ngram: str, cand_tid: int, original_text: str
    ) -> Optional[Union[str, float]]:
        """Strategy A: In-line mapping (Value is in the same text block as the Key)."""
        if cand_tid not in self.text_data_buffer or not self.text_data_buffer[cand_tid].strip():
            return None

        parts = re.split(re.escape(field_ngram), original_text, flags=re.IGNORECASE)
        cand_text = clean_string(parts[-1]) if len(parts) > 1 else ''

        if cand_text:
            cand_start_ngrams = self._process_text_ngrams(
                cand_text, n_max=6, include_full=True, start=True, reverse=True
            )
            parsed = self._extract_and_consume_value(field_name, cand_start_ngrams, cand_tid, 'A')
            if isinstance(parsed, str):
                return parsed.strip(' :#,-')
            elif parsed:
                return parsed

        return None

    def _extract_value_from_group(
        self,
        field_name: str,
        key_tid: int,
        key_group: int,
        texts: List[DoclingText],
        candidates: List[List[Tuple[int, List[str]]]],
    ) -> Optional[Union[str, float]]:
        """Strategy B: Spatial mapping (Value is in a separate text block)."""
        key_bbox = get_bbox(texts[key_tid])
        if not key_bbox:
            return None

        # Search within the SAME group defined by Docling
        group_cands = []
        for neighbor_tid, _ in candidates[key_group]:
            if neighbor_tid == key_tid or neighbor_tid not in self.text_data_buffer:
                continue

            neighbor_text = texts[neighbor_tid]
            neighbor_bbox = get_bbox(neighbor_text)
            if not neighbor_bbox:
                continue

            is_horz = is_horizontally_aligned(key_bbox, neighbor_bbox)
            is_vert = is_vertically_aligned(key_bbox, neighbor_bbox)

            # Look for values to the RIGHT or BELOW the key - origin is BOTTOMLEFT
            is_right = is_horz and neighbor_bbox.l >= key_bbox.l
            is_below = is_vert and neighbor_bbox.t <= key_bbox.t

            if is_right or is_below:
                dist = euclidean_dist(key_bbox, neighbor_bbox)
                group_cands.append((self.text_data_buffer[neighbor_tid], dist, neighbor_tid))

        # Sort by proximity
        group_cands.sort(key=lambda x: x[1])
        logger.debug(
            f'    - Strategy B shortlisted: {len(group_cands)} of group #{key_group}: {[c[0] for c in group_cands]}'
        )
        for cand_text, _, cand_tid in group_cands:
            cand_ngrams = self._process_text_ngrams(cand_text, n_max=6, include_full=True, start=True, reverse=True)
            parsed = self._extract_and_consume_value(field_name, cand_ngrams, cand_tid, 'B')
            if isinstance(parsed, str):
                return parsed.strip(' :#,-')
            elif parsed:
                return parsed
        return None

    def _get_missing_fields(
        self,
        mapped_data: Dict[str, Union[str, float, None]],
        ignore_types: Set['str'] = {'amount', 'rate'},
        reverse: bool = True,
        swap_fields: List[Tuple[str]] = C_SWAP_FILEDS,
    ) -> List[str]:

        missing_fields = [
            f for f, t in self.field_value_types.items() if (f not in mapped_data and t not in ignore_types)
        ]
        # Process missing fields in reverse order
        if reverse is True:
            missing_fields.reverse()

        # Reorder orphan fields for optimal Strategy C heuristic result
        if swap_fields:
            missing_fields = self._swap_field_pairs(swap_fields, missing_fields)

        return missing_fields

    def _extract_orphan_value(self, field_name: str) -> Optional[Union[str, float]]:
        """Strategy C: Orphan Remainder Recovery from unused text buffer."""
        for tid, remaining_text in self.text_data_buffer.items():
            if not remaining_text:
                continue
            cand_ngrams = self._process_text_ngrams(remaining_text, n_max=6, include_full=True, reverse=True)
            parsed = self._extract_and_consume_value(field_name, cand_ngrams, tid, 'C')
            if isinstance(parsed, str):
                return parsed.strip(' :#,-')
            elif parsed:
                return parsed
        return None

    @staticmethod
    def _add_orphan_field_value(field, value, mapped_data: Dict[str, Union[str, float, None]]):
        if 'total' in mapped_data and value == mapped_data['total']:
            logger.debug(f"  Skipping orphan '{value}' for '{field}' - duplicate of already extracted total")
            return
        elif field.endswith('addr') and any(value == mapped_data.get(f, '') for f in ['issuer_addr', 'receiver_addr']):
            logger.debug(f"  Skipping orphan '{value}' for '{field}' - duplicate of already extracted address")
            return

        mapped_data[field] = value

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
            loc_details = [f'group[{locations[i][0]}]: text[{locations[i][1]}]' for i in range(len(locations))]

            logger.debug(
                f'\n  - Field "{field_name}" matched key "{best_ngram}" (score: {score:.2f} [{metric}])'
                f' at {len(locations)} locations: {", ".join(loc_details)}'
            )

            if len(match['candidates']) > 1:
                alt_cands = [f"'{c[0]}' ({c[1]:.2f} [{c[2]}])" for c in match['candidates']]
                logger.debug(f'    - Candidates considered:\n    {", ".join(alt_cands)}')

        # Log unmatched fields that were below threshold but were "candidates"
        field_names = list([f.alias for f in self.fields])
        unmatched = [
            f for f in field_names if f not in ngram_match_by_field or ngram_match_by_field[f]['best'][1] < threshold
        ]
        if unmatched:
            logger.debug(f'\n  => Unmatched Fields: {", ".join(unmatched)}\n')

    def extract_key_candidates(self, doc: DoclingDoc) -> List[List[Tuple[int, List[str]]]]:
        """Extract key candidates from doc groups and orphan texts."""
        candidates: List[List[Tuple[int, List[str]]]] = []

        groups = doc.groups
        texts = doc.texts

        # Consider all groups
        processed_texts: Set[int] = set()
        for group in groups:
            group_texts = []
            children = group.children
            for child in children:
                t_obj = resolve_text(child, doc, texts)
                if t_obj:
                    group_texts.append(t_obj)
                    processed_texts.add(id(t_obj))

            if group_texts:
                candidates.append(self._process_key_candidate_ngrams(group_texts))

        # Consider all other orphans texts
        orphan_texts = []
        for text in texts:
            if id(text) not in processed_texts:
                orphan_texts.append(text)

        # Treat orphans as an extra group
        if orphan_texts:
            candidates.append(self._process_key_candidate_ngrams(orphan_texts))

        return candidates

    def map_field_keys(
        self,
        candidates: List[List[Tuple[int, List[str]]]],
        threshold: float = 0.60,
        lex_threshold: float = 0.85,
    ) -> Dict[str, Tuple[str, List[Tuple[int, int]]]]:

        # Map each n-gram to a list of its locations to prevent overwriting
        ngrams_map: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for g, candidates_group in enumerate(candidates):
            for text_id, text_ngrams in candidates_group:
                for ngram in text_ngrams:
                    ngrams_map[ngram].append((g, text_id))

        matches_by_field = map_labels_to_fields(
            labels=ngrams_map.keys(),
            smapper=self.smapper,
            lmapper=self.lmapper,
            lexical_threshold=lex_threshold,
            lexical_rank=2,
        )

        if logger.isEnabledFor(logging.DEBUG):
            self._log_key_mapping_debug_results(matches_by_field, ngrams_map, threshold)

        fields_map = {
            field: (matches_by_field[field]['best'][0], ngrams_map[matches_by_field[field]['best'][0]])
            for field in self.field_value_types.keys()
            if (matches_by_field.get(field) is not None and matches_by_field[field]['best'][1] >= threshold)
        }
        return fields_map

    def map_field_values(
        self,
        doc: DoclingDoc,
        fields_map: Dict[str, Tuple[str, List[Tuple[int, int]]]],
        candidates: List[List[Tuple[int, List[str]]]],
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
            fields_map: Map of field_name keys to (n-gram, List[(group_id, text_id)])
            candidates: Nested list of (text_id, n-grams) grouped by spatial relevance

        Returns:
            Dict mapping field names to parsed values
        """
        texts = getattr(doc, 'texts', [])
        summary_data: Dict[str, Union[str, float, None]] = {}

        # Text buffer to keep track of remaining text after key/value substring consumption
        self._initialize_text_buffer(doc, fields_map)

        # Match values for each identified field
        logger.debug('Resolving field values (Strategy A: same-text RHS; B: Neighbour texts in group):')

        empty_fields_by_name = {}
        for field_name, (field_ngram, locations) in fields_map.items():
            logger.debug(f"Field '{field_name}' (Key: '{field_ngram}')")

            value_match: Union[str, float, None] = None
            for g, tid in locations:
                # Strategy A: In-line mapping
                value_match = self._extract_value_from_text(field_name, field_ngram, tid, texts[tid].text)

                # Strategy B: Spatial mapping
                if value_match is None:
                    value_match = self._extract_value_from_group(field_name, tid, g, texts, candidates)

                # If field value is found near current key location, stop checking for values in other key locations
                if value_match is not None:
                    if field_name.startswith('tax_'):
                        value_match = self._process_tax_field(field_name, field_ngram, value_match, summary_data)
                    summary_data[field_name] = value_match
                    break

            if value_match is None:
                empty_fields_by_name[field_name] = (field_ngram, locations)
                logger.debug(
                    f'    - No value found across {len(locations)} key locations:'
                    f' [{[(texts[loc[1]].text, f"g{loc[0]}: {candidates[loc[0]]}") for loc in locations]}]'
                )

        # Final Sweep: Extract non-numeric values from buffer if they fit missing fields in order
        missing_fields = self._get_missing_fields(summary_data)
        if missing_fields:
            logger.debug(f'Final Sweep: Attempting orphan-value recovery for missing fields: {missing_fields}')
            for field_name in missing_fields:
                logger.debug(f"Field '{field_name}':")

                anchor_field = None
                for anchor_name in {'issuer', 'receiver'}:
                    if anchor_name in summary_data and anchor_name in field_name:
                        anchor_field = anchor_name
                        break

                parsed = None
                if anchor_field:
                    for g, tid in fields_map[anchor_field][1]:
                        parsed = self._extract_value_from_group(field_name, tid, g, texts, candidates)

                if parsed is None:
                    # Strategy C: Greedy-match orphan field relying on order heuristics
                    parsed = self._extract_orphan_value(field_name)

                if parsed:
                    self._add_orphan_field_value(field_name, parsed, summary_data)

        return summary_data

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
            device = check_gpu_compatibility(device)

        if smapper is None:
            smapper = SemanticMapper.create(fields, device=device)
        if lmapper is None:
            lmapper = LexicalMapper.create(fields)
        if ner is None:
            ner = spacy.load('en_core_web_md')

        return cls(fields, smapper, lmapper, ner, threshold)


def extract_invoice_summary(
    doc: DoclingDoc,
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

    candidates = extractor.extract_key_candidates(doc)
    fields_by_key = extractor.map_field_keys(candidates)

    mapped_data = extractor.map_field_values(doc, fields_by_key, candidates)

    # Pydantic handles the final validation
    return InvoiceSummary.model_validate(mapped_data)
