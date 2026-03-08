# IDP-CV

## Overview
IDP-CV (Intelligent Document Processing - Computer Vision) is a Python pipeline designed to extract structured information from unstructured documents. The project leverages modern ML and NLP frameworks to parse documents, identify text elements, and map them to strictly typed schemas. 

Key technologies include:
- **Docling**: For robust document parsing and layout analysis.
- **SentenceTransformers**: For semantic similarity mapping (defaulting to Granite small embeddings).
- **Spacy**: For Sequence Tagging with Named Entity Recognition.
- **PyTorch & ONNX Runtime**: For underlying model inference.
- **Pydantic**: For strict data validation and schema definition.

## Setup and Installation

### Standard CPU Installation
The default installation is optimized for CPU environments to ensure maximum compatibility and straightforward deployment.

1. Clone the repository and navigate to the project root:
   ```bash
   git clone git@github.com:krus7ev/idp-cv.git
   cd idp-cv
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the package and its dependencies:
   ```bash
   pip install -e .
   ```
   *Note: This process will automatically download the necessary HuggingFace and Spacy models required for local CPU inference.*

### Docker Deployment
The pipeline can also be run via Docker without managing host dependencies:
```bash
docker build -t idp-cv:latest .
docker run -v $(pwd)/data:/app/data idp-cv:latest
```

### GPU Usage (General & Legacy Setup)
For environments requiring hardware acceleration, one needs to replace the CPU-bound pip dependencies with their system's specific CUDA distributions.

**Legacy Setup Example (CUDA 11.8):**
The project was developed on older hardware which specifically required using PyTorch 2.2.2 with a CUDA 11.8 build to support both the latest Docling version and work (by backwards-compatibility) with the 470 (CUDA 11.4) driver installed. For a similar setup, these specific dependency versions can be enforced by providing a `constraints.txt` file like the example one during the initial installation:

```bash
pip install -e . -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu118
```
Depending on the GPU architecture, this may vary or not be necessary.

## Usage

### Interactive Notebook
For exploration and step-by-step debugging, use the provided Jupyter Notebook: `idp_cv_invoices.ipynb`. It walks through the initialization of models, parsing of document layouts, and the granular extraction steps mapping to Pydantic schemas natively.

### Command Line Script
For batch processing, a dedicated runner script (`run_extraction.py`) handles inputs iteratively and outputs structured JSON schemas mapping values to the designated extracted fields.
```bash
python run_extraction.py --input data/example_invoices/file_inputs --output data/example_invoices/idp-cv_outputs
```
*(Pass `-h` or `--help` to view all available arguments directly from the script).*

### Docker Container
When running via Docker, the pipeline executes the CLI script automatically via its entrypoint (`run_extraction.py`). Mount a local data directory to the container to safely pass inputs and retrieve outputs:
```bash
# Map local ./data folder to the container's /app/data
docker run -v $(pwd)/data:/app/data idp-cv:latest --input /app/data/example_invoices/file_inputs --output /app/data/example_invoices/idp-cv_outputs
```
Any command line arguments appended to `docker run` are passed directly into the Python runner script.

## Configuring Schemas

The extraction targets are strictly defined using Pydantic schemas located in `idp_cv/schemas.py`. By defining a schema, the pipeline is instructed on exactly what fields to search for, their expected custom *value-types*, and whether they are required.

### Default Invoice Schema
Currently, the pipeline defaults to an invoice extraction profile. The default schemas `InvoiceSummary` and `TableLine` are configured to extract standard billing metadata, such as:
- **Invoice Number** & **Dates** (Issue date, Due date)
- **Vendor & Client Information** (Names, Addresses, Tax IDs)
- **Financial Totals** (Net amount, Tax amount, Gross total)
- **Line Items** (Description, Quantity, Unit Price, Line Total)

### Customizing Extraction
**Important Caveat**: Currently, the Pydantic schemas are tightly coupled to deterministic logic within `extract.py` and `parse.py`. The models use specific heuristics targeting specific field names (like `invoice_number` or `net_amount`) rather than just the more general `value_type`s from each schema field's `json_schema_extra`. 

While custom schemas can be defined like the example below, the pipeline's extraction logic does not blindly parse arbitrary fields. The current mapping mechanism will need to be generalized further or formalized through a configuration-driven schema in order for the system to support completely arbitrary document schemas natively.

```python
# Conceptual arbitrary schema implementation
from pydantic import BaseModel, Field
from typing import Optional

class CustomDocumentSchema(BaseModel):
    client_name: str = Field(
        default=None,
        description="The name of the client the document is issued to.",
        alias=doc_title
        json_schema_extra={
            'aliases': ['client:', 'customer:', 'client_name:', 'customer name:',] 
            'value_type': 'name'
        }
    )
    reference_id: Optional[str] = Field(default=None, description="Alphanumeric reference ID" ...
```

Once parsing heuristics are fully generalized to support automated field mapping dynamically, custom schemas could be passed to the runner script to capture dynamic data constraints automatically.