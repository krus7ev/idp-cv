import argparse
import json
import logging
import os
import sys
import traceback
import warnings
from pathlib import Path

import pandas as pd
import spacy
import torch
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, ImageFormatOption, InputFormat, PdfFormatOption


# Ensure models are downloaded before switching to offline mode
def _ensure_models_downloaded():
    try:
        import spacy.util
        from sentence_transformers import SentenceTransformer

        from idp_cv.constants import DEFAULT_GRANITE_MODEL_ID

        missing = False
        if not spacy.util.is_package('en_core_web_md'):
            missing = True
        else:
            orig_offline = os.environ.get('TRANSFORMERS_OFFLINE', '0')
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            try:
                # Try loading offline to see if it's cached
                SentenceTransformer(DEFAULT_GRANITE_MODEL_ID)
            except Exception:
                missing = True
            finally:
                os.environ['TRANSFORMERS_OFFLINE'] = orig_offline

        if missing:
            logging.info('Models not found via offline check. Attempting download...')
            import subprocess

            subprocess.check_call([sys.executable, 'download_models.py'])
    except Exception as e:
        logging.warning(f'Failed to check or download models: {e}')


_ensure_models_downloaded()

# Force offline mode for transformers/sentence-transformers
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from idp_cv.constants import SUPPORTED_FORMATS
from idp_cv.extract import DocumentFieldExtractor, extract_invoice_summary
from idp_cv.parse import (
    LexicalMapper,
    SemanticMapper,
    map_table_to_line_items,
)
from idp_cv.schemas import InvoiceSummary, TableLine
from idp_cv.utils import draw_page_items, extract_clean_table_data

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout, force=True)
logging.getLogger('RapidOCR').setLevel(logging.WARNING)
logging.getLogger('docling').setLevel(logging.WARNING)
logging.getLogger('idp_cv').setLevel(logging.DEBUG)
warnings.filterwarnings('ignore', category=UserWarning, module='docling')


def process_and_extract_results(
    input_dir: Path, output_dir: Path, device: str = 'cpu', do_summary: bool = True, do_viz: bool = True
):
    """
    Unified processing routine for docling conversion, table mapping and summary extraction.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pipeline and Converter init
    # NOTE: Using 'onnx' backend for faster CPU inference
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=device),
        generate_page_images=do_viz,
        images_scale=2.0 if do_viz else 1.0,
        do_ocr=True,
        ocr_options=RapidOcrOptions(backend='onnxruntime', print_verbose=False),
        do_table_structure=True,
        force_backend_text=False,
        generate_parsed_pages=True,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )

    # Initialize table and summary schemas
    table_schema_fields = list(TableLine.model_fields.values())
    summary_schema_fields = list(InvoiceSummary.model_fields.values())

    # Pre-initialize Mappers/NER
    table_lex_mapper = LexicalMapper.create(table_schema_fields)
    summary_lex_mapper = LexicalMapper.create(summary_schema_fields)

    table_sem_mapper = SemanticMapper.create(table_schema_fields, device=device)
    summary_sem_mapper = SemanticMapper.create(summary_schema_fields, device=device)

    ner = spacy.load('en_core_web_md')

    extractor = DocumentFieldExtractor.create(
        schema=InvoiceSummary,
        smapper=summary_sem_mapper,
        lmapper=summary_lex_mapper,
        ner=ner,
        device=device,
        threshold=0.5,
    )

    # Execution
    files_to_process = sorted([f for f in input_dir.glob('*') if f.suffix.lower() in SUPPORTED_FORMATS])

    if not files_to_process:
        print(f'No supported documents found in {input_dir}')
        return []

    print(f'Found {len(files_to_process)} files to process in {input_dir}')
    conv_results = converter.convert_all(files_to_process)

    final_results = []

    for result in conv_results:
        if result.status != 'success':
            print(f'Failed: {result.input.file.name}')
            continue

        doc = result.document
        stem = result.input.file.stem
        doc_entry = {'Document': stem}

        # Export raw JSON
        output_json = output_dir / f'{stem}.json'
        with open(output_json, 'w') as f:
            json.dump(doc.export_to_dict(), f)

        # Visualisation
        if do_viz:
            for page_no, page_item in doc.pages.items():
                if page_item.image is None:
                    continue
                img = page_item.image.pil_image.convert('RGB')
                draw_page_items(img, doc.texts, page_no, page_item, 'blue', 2, True)
                draw_page_items(img, doc.tables, page_no, page_item, 'magenta', 4, True)
                img.save(output_dir / f'viz_{stem}_p{page_no}.png')

        # Table Mapping
        doc_entry['tables'] = extract_clean_table_data(doc)
        if doc_entry['tables']:
            items_table = [doc_entry['tables'][0]]  # NOTE Assuming the first table is the items table
        else:
            items_table = []

        if len(doc_entry['tables']) >= 2:
            summary_table = [doc_entry['tables'][1]]  # NOTE Assuming the second table is the summary table
        else:
            summary_table = []

        if items_table:
            doc_entry['line_items'] = map_table_to_line_items(
                items_table, smapper=table_sem_mapper, lmapper=table_lex_mapper
            )
        if summary_table:
            doc_entry['summary_table'] = map_table_to_line_items(
                summary_table, smapper=table_sem_mapper, lmapper=table_lex_mapper
            )

        # Summary Extraction
        if do_summary:
            try:
                summary = extract_invoice_summary(doc, extractor=extractor)
                doc_entry['summary'] = summary.model_dump(exclude_none=False)
            except Exception as e:
                print(f'[{stem}] Summary error: {e}')
                traceback.print_exc()

        if 'summary_table' in doc_entry:
            for field in ['total_amount', 'net_amount', 'tax_amount', 'tax_rate', 'shipping_cost']:
                if 'summary' in doc_entry and not getattr(summary, field, None):
                    for summary_row in doc_entry['summary_table'][0]:
                        field_value = getattr(summary_row, field)
                        if field_value:
                            if field == 'tax_rate' and isinstance(field_value, float) and field_value > 1.0:
                                field_value /= 100.0
                            doc_entry['summary'][field] = field_value

        final_results.append(doc_entry)
        print(f'Completed: {stem}')

    return final_results


def main():
    parser = argparse.ArgumentParser(description='Extract invoice data using IDP-CV')
    parser.add_argument(
        '-i',
        '--input',
        type=Path,
        default=Path('./data/example_invoices/file_inputs/'),
        help='Input directory containing invoice documents (PDF, images)',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=Path('./data/example_invoices/idp-cv_outputs/'),
        help='Output directory for generated JSON, visualizations, and summary report',
    )
    parser.add_argument('--no-viz', action='store_true', help='Disable generation of visualization images')
    parser.add_argument(
        '--no-summary', action='store_true', help='Disable extraction of top-level invoice summary fields'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use for embeddings/OCR. Defaults to cuda:0 if available, else cpu.',
    )
    parser.add_argument('--batch-concurrency', type=int, default=4, help='Document batch concurrency for Docling')
    parser.add_argument('--batch-size', type=int, default=4, help='Document batch size for Docling')

    args = parser.parse_args()

    # Determine Device
    device = args.device
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f'Using device: {device}')

    # docling runtime settings
    settings.perf.doc_batch_concurrency = args.batch_concurrency
    settings.perf.doc_batch_size = args.batch_size

    # Run extraction
    results = process_and_extract_results(
        input_dir=args.input,
        output_dir=args.output,
        device=device,
        do_summary=not args.no_summary,
        do_viz=not args.no_viz,
    )

    # Post processing and formatting
    summary_data = []
    for res in results:
        if 'summary' in res:
            entry = res['summary'].copy()
            entry['Document'] = res['Document']
            summary_data.append(entry)

    print('\n--- Processing Complete ---')
    if summary_data:
        df = pd.DataFrame(summary_data)
        cols = ['Document'] + [c for c in df.columns if c != 'Document']
        df = df[cols].set_index('Document')

        # Save overarching summary to CSV
        summary_csv = args.output / 'invoices_summary.csv'
        df.to_csv(summary_csv)
        print(f'Summary saved to {summary_csv}')
    else:
        print('No summary results generated.')


if __name__ == '__main__':
    main()
