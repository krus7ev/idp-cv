import os

import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from idp_cv.constants import DEFAULT_GRANITE_MODEL_ID


def download_models():
    print('Downloading models for offline usage...')

    # 1. Download Spacy model (if not already installed via requirements)
    # The direct URL in requirements.txt should handle this, but verify load
    try:
        print("Checking Spacy model 'en_core_web_md'...")
        spacy.load('en_core_web_md')
        print('✓ Spacy model loaded successfully.')
    except OSError:
        print('Spacy model not found. Installing...')
        os.system('python -m spacy download en_core_web_md')

    # 2. Download SentenceTransformer model
    model_name = DEFAULT_GRANITE_MODEL_ID
    print(f"Downloading SentenceTransformer model '{model_name}'...")

    # This downloads the model to the default cache directory (~/.cache/huggingface/hub)
    # allowing subsequent offline loading with local_files_only=True
    SentenceTransformer(model_name)
    print('✓ SentenceTransformer model downloaded successfully.')

    # 3. Explicitly verify HuggingFace Transformers components if needed separately
    # (SentenceTransformer handles this internally usually, but just in case)
    print(f"Verifying Transformers components for '{model_name}'...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModel.from_pretrained(model_name)
    print('✓ Transformers components verified.')

    print('\nAll models are ready for offline usage!')


if __name__ == '__main__':
    download_models()
