import logging
import re
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

import torch
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from sentence_transformers import SentenceTransformer, util

from .constants import DEFAULT_GRANITE_MODEL_ID, RE_IS_PURE_NUMERIC
from .schemas import TableLine, ValidatedTableLine
from .types import (
    FieldMappingResult,
)
from .utils import (
    load_embedding_model,
    normalize_text,
)

logger = logging.getLogger(__name__)


class LexicalMapper:
    """Lexical field matcher using Levenshtein similarity over field aliases."""

    def __init__(self, fields: Sequence[FieldInfo]):
        self.fields = fields
        self.field_aliases: Dict[str, List[str]] = {}
        self.set_aliases()

    def set_aliases(self) -> None:
        """Precompute normalized aliases for all fields."""
        for field in self.fields:
            aliases = []
            if field.json_schema_extra:
                aliases = field.json_schema_extra.get('aliases', [])

            if not aliases:
                fallback = field.title or field.alias
                aliases = [fallback] if fallback else []

            if field.alias and aliases:
                self.field_aliases[field.alias] = [normalize_text(alias) for alias in aliases]

    @property
    def all_aliases(self) -> List[str]:
        """Return a flattened list of all unique aliases for all fields."""
        unique = set()
        for aliases in self.field_aliases.values():
            unique.update(aliases)
        return list(unique)

    @staticmethod
    def similarity(a: str, b: str) -> float:
        """Compute Levenshtein similarity between two normalized strings."""
        a_norm, b_norm = normalize_text(a), normalize_text(b)
        if not a_norm and not b_norm:
            return 1.0
        if not a_norm or not b_norm:
            return 0.0

        n, m = len(a_norm), len(b_norm)
        prev_row = list(range(m + 1))
        for i in range(1, n + 1):
            curr_row = [i] + [0] * m
            for j in range(1, m + 1):
                cost = 0 if a_norm[i - 1] == b_norm[j - 1] else 1
                curr_row[j] = min(
                    prev_row[j] + 1,
                    curr_row[j - 1] + 1,
                    prev_row[j - 1] + cost,
                )
            prev_row = curr_row

        distance = prev_row[m]
        return 1.0 - (distance / max(n, m))

    def map_field(self, source_label: str, rank: int = 1) -> Union[tuple[float, str], List[tuple[float, str]]]:
        """Find best matching field(s) for source label using lexical similarity."""
        if not self.field_aliases:
            return None, 0.0

        source_norm = normalize_text(source_label)

        results: List[tuple[float, str]] = []
        for field_name, aliases in self.field_aliases.items():
            for alias in aliases:
                score = self.similarity(source_norm, alias)
                results.append((field_name, score))

        results = sorted(results, key=lambda x: x[1], reverse=True)

        return results[:rank]

    @classmethod
    def create(cls, schema) -> Optional['LexicalMapper']:
        if not schema or (
            not (isinstance(schema, type) and issubclass(schema, BaseModel))
            and not (isinstance(schema, Sequence) and isinstance(schema[0], FieldInfo))
        ):
            return
        fields = schema if isinstance(schema, Sequence) else list(schema.model_fields.values())

        return cls(fields)


class SemanticMapper:
    """Semantic field matcher using embedding-based similarity."""

    def __init__(self, fields: Sequence[FieldInfo], model: SentenceTransformer, normalize: bool):
        """Initialize mapper with field schema."""
        self.fields = fields
        self.model = model
        self.normalize = normalize

        self.field_embeddings: Dict[str, List[torch.Tensor]] = {}
        self.set_embeddings()

    def get_embedding(self, text: str) -> torch.Tensor:
        """Compute embedding vector for input string."""
        return self.model.encode(text, convert_to_tensor=True, normalize_embeddings=self.normalize)

    def set_embeddings(self) -> None:
        """Precompute embeddings for all of field's aliases or its title"""
        for field in self.fields:
            aliases = []
            if field.json_schema_extra:
                aliases = field.json_schema_extra.get('aliases', [])
            if not aliases and field.title:
                self.field_embeddings[field.alias] = [self.get_embedding(field.title)]
            else:
                self.field_embeddings[field.alias] = [self.get_embedding(alias) for alias in aliases]

    @staticmethod
    def normalize_sim(score: float) -> float:
        """Normalize cosine similarity from [-1, 1] to [0, 1]."""
        return (score + 1.0) / 2.0

    def map_field(self, source_label: str) -> tuple[Optional[str], float]:
        """Find best matching field for the source label (header) using cosine similarity."""
        if not self.field_embeddings:
            return None, -1.0

        source_embedding = self.get_embedding(source_label)

        alias_embeddings: List[torch.Tensor] = []
        fields_idx: List[str] = []
        for field_name, alias_vecs in self.field_embeddings.items():
            if not alias_vecs:
                continue
            alias_embeddings.extend(alias_vecs)
            fields_idx.extend([field_name] * len(alias_vecs))

        if not alias_embeddings:
            return None, -1.0

        similarities = util.cos_sim(source_embedding, torch.stack(alias_embeddings))[0]
        best_alias_id = similarities.argmax().item()

        return fields_idx[best_alias_id], self.normalize_sim(similarities[best_alias_id].item())

    @classmethod
    def create(cls, schema, model_or_id=None, normalize=False, device='cpu') -> Optional['SemanticMapper']:
        if not schema or (
            not (isinstance(schema, type) and issubclass(schema, BaseModel))
            and not (isinstance(schema, Sequence) and isinstance(schema[0], FieldInfo))
        ):
            return
        fields = schema if isinstance(schema, Sequence) else list(schema.model_fields.values())

        device = device or 'cpu'
        model, model_id = None, DEFAULT_GRANITE_MODEL_ID
        if model_or_id:
            if isinstance(model_or_id, str):
                model_id = model_or_id
            elif isinstance(model_or_id, SentenceTransformer):
                model = model_or_id

        if model is None:
            model, _ = load_embedding_model(model_id=model_id, device=device)

        return cls(fields, model, normalize)


def map_labels_to_fields(
    labels: Sequence[str],
    smapper: SemanticMapper,
    lmapper: LexicalMapper,
    lexical_threshold: float = 0.8,
    field_value_types: Optional[Dict[str, str]] = None,
    label_samples: Optional[Dict[str, List[str]]] = None,
    validator_func: Optional[Callable[[str, str, Dict[str, str]], Union[str, float, None]]] = None,
    semantic_threshold: float = 0.98,
    lexical_rank=1,
) -> Dict[str, FieldMappingResult]:
    """
    Map a list of labels (e.g., table columns or n-grams) to schema fields
    Returns: {
        field_name: {
            'best': (label, score, metric),
            'candidates': [(label, score, metric), ...]
        }
    }
    """
    all_matches_by_field: Dict[str, List[tuple[str, float, str]]] = defaultdict(list)

    for label in labels:
        # High lexical similarity above threshold takes priority
        lex_mapped = lmapper.map_field(label, rank=lexical_rank)
        for lex_match, lex_score in lex_mapped:
            if lex_match and lex_score >= lexical_threshold:
                match_field, match_score, metric = lex_match, lex_score, 'lexical'
            else:
                sem_match, sem_score = smapper.map_field(label)
                if sem_match and lex_match:
                    if sem_score >= semantic_threshold and sem_score >= lex_score:
                        match_field, match_score, metric = sem_match, sem_score, 'semantic'
                    else:
                        match_field, match_score, metric = lex_match, lex_score, 'lexical'
                elif sem_match and sem_score >= semantic_threshold:
                    match_field, match_score, metric = sem_match, sem_score, 'semantic'
                elif lex_match:
                    match_field, match_score, metric = lex_match, lex_score, 'lexical'
                else:
                    continue

            # Optional value-type validation
            if validator_func and field_value_types and label_samples:
                v_type = field_value_types.get(match_field)
                samples = label_samples.get(label, [])
                if v_type and samples:
                    valid_count = 0
                    testable_samples = [s for s in samples if s.strip()]
                    if testable_samples:
                        for s in testable_samples:
                            if validator_func(match_field, s, field_value_types) is not None:
                                valid_count += 1

                        if valid_count / (len(testable_samples) or 1) < 0.5:
                            continue

            all_matches_by_field[match_field].append((label, match_score, metric))

    # Resolve best match per field and prepare response
    result: Dict[str, FieldMappingResult] = {}
    used_labels = []
    for field_name, candidates in all_matches_by_field.items():
        # Sort candidates: Lexical matches > Semantic matches, then by score
        sorted_cands = sorted(candidates, key=lambda x: (1 if x[2] == 'lexical' else 0, x[1]), reverse=True)
        final_cand = (
            sorted_cands[0] if len(sorted_cands) == 1 or sorted_cands[0][0] not in used_labels else sorted_cands[1]
        )
        result[field_name] = {'best': final_cand, 'candidates': sorted_cands}
        used_labels.append(final_cand[0])

    return result


def validate_column_value(
    field_name: str, value_text: str, field_value_types: Dict[str, str]
) -> Union[str, float, None]:
    """Test if column data is compatible with column header's value type"""
    v_type = field_value_types.get(field_name, 'string')
    val = value_text.strip()
    if not val:
        return None

    # Check for essentially numeric content
    is_pure_numeric = bool(re.fullmatch(RE_IS_PURE_NUMERIC, val))
    if v_type in ('amount', 'rate'):
        # For numeric fields, we want a valid number
        return TableLine.parse_numeric(val)

    if v_type in ('string', 'name', 'address') and is_pure_numeric:
        # For string fields, we reject if it's purely numeric
        return None

    return val


def map_table_to_line_items(
    clean_tables: List[List[Dict]],
    smapper: Optional[SemanticMapper] = None,
    lmapper: Optional[LexicalMapper] = None,
    device: str = 'cpu',
    threshold: float = 0.3,
    lexical_threshold: float = 0.75,
    model_schema: Type[ValidatedTableLine] = TableLine,
) -> List[List[BaseModel]]:
    """Map table rows to ValidatedTableLine objects with lexical-first matching."""

    fields = list(model_schema.model_fields.values())
    field_value_types = {
        f.alias: f.json_schema_extra.get('value_type', 'string') for f in fields if f.json_schema_extra
    }
    if smapper is None:
        smapper = SemanticMapper.create(fields, device=device)
    if lmapper is None:
        lmapper = LexicalMapper.create(fields)

    all_tables_lines: List[List[BaseModel]] = []
    for table_rows in clean_tables:
        if not table_rows:
            all_tables_lines.append([])
            continue

        # Prepare column data samples (first 10 rows)
        label_samples = {}
        for col in table_rows[0].keys():
            label_samples[col] = [row.get(col, '') for row in table_rows[:10]]

        logger.debug(f'Mapping table with columns: {list(table_rows[0].keys())}')
        logger.debug(f'Using LexicalMapper with aliases: {lmapper.field_aliases}')
        logger.debug(f'Using SemanticMapper with fields: {list(smapper.field_embeddings.keys())}')
        logger.debug(f'Sample data for validation: { {col: label_samples[col][:3] for col in label_samples} }')

        matches_by_field = map_labels_to_fields(
            labels=list(table_rows[0].keys()),
            smapper=smapper,
            lmapper=lmapper,
            lexical_threshold=lexical_threshold,
            field_value_types=field_value_types,
            label_samples=label_samples,
            validator_func=validate_column_value,
        )

        logger.debug(f'Mapping results: {matches_by_field}')

        # Build 1-to-1 mapping from source column to schema field
        # Prioritize strongest field-to-column match if multiple fields compete for the same column
        column_map = {}
        column_scores = {}
        for match_field, res in matches_by_field.items():
            source_col, score, _ = res['best']
            if score >= threshold:
                if source_col not in column_scores or score > column_scores[source_col]:
                    column_map[source_col] = match_field
                    column_scores[source_col] = score

        table_lines: List[BaseModel] = []
        for row in table_rows:
            mapped_row = {column_map[k]: v for k, v in row.items() if k in column_map}

            try:
                table_lines.append(model_schema(**mapped_row))
                logger.debug(f'Mapped row: {mapped_row} to table line: {table_lines[-1]}.')
            except Exception as exc:
                logger.error(f'Skipping row due to validation error: {exc}')

        all_tables_lines.append(table_lines)
        logger.debug(f'Mapped {len(table_lines)} lines for current table.')

    return all_tables_lines
