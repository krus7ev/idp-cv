import re
from typing import NamedTuple

SUPPORTED_FORMATS = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff']

# Validation CONSTANTS
CURRENCY_MAP = {
    'USD': '$',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'INR': '₹',
    'RUB': '₽',
    'ILS': '₪',
    'KRW': '₩',
    'VND': '₫',
    'AZN': '₼',
    'THB': '฿',
    'NGN': '₦',
    'CHF': '₣',
    'PYG': '₲',
    'BTC': '₿',
    'XEU': '₠',
    'BRZ': '₢',
    'ITL': '₤',
    'MIL': '₥',
    'ESP': '₧',
    'PKR': '₨',
    'PHP': '₱',
    'MNT': '₮',
    'LAK': '₭',
    'GRD': '₯',
    'PFN': '₰',
    'ARA': '₳',
    'LVT': '₶',
    'SPM': '₷',
    'KZT': '₸',
}

CURRENCY_TOKENS = tuple(set(CURRENCY_MAP) | set(CURRENCY_MAP.values()))


CURRENCY_RE = re.compile('|'.join(re.escape(t) for t in sorted(CURRENCY_TOKENS, key=len, reverse=True)))
FIRST_DIGIT_RE = re.compile(r'\d')
NUMERIC_CLEAN_RE = re.compile(r'[^\d,\.\s]')
SPACE_RE = re.compile(r'\s+')
VALID_NUMBER_RE = re.compile(r'\d+(?:\.\d+)?')
SIGNED_INT_SPACED_RE = re.compile(r'^\s*[+-]?\s*\d+\s*$')

RE_IS_PURE_NUMERIC = re.compile(r'[\d\s\.,\$\€\%\-\(\)]+')
RE_ID_ALPHANUMERIC = r'[A-Z0-9\-_./#\s]{2,40}'
RE_ADDR_ALPHANUMERIC = r'[A-Z0-9\s.,\-#\'/&()]+'
RE_DOUBLE_DIGIT_PERCENTAGE = r'(?<!\d)\d{1,2}%(?!\d)'

COMPANY_SUFFIXES = {'limited', 'llc', 'ltd', 'inc', 'gmbh', 'corp'}
ADDR_NER_TAGS = {'LOC', 'FAC', 'CARDINAL', 'GPE', 'ORG', 'PERSON', 'DATE'}
LOC_NER_TAGS = {'LOC', 'FAC', 'GPE'}
NAME_NER_TAGS = {'PERSON', 'NORP', 'ORG', 'GPE'}


class SummaryExtractionConfig(NamedTuple):
    amount_aliases: set = {'total', 'subtotal', 'tax', 'shipping'}
    party_name_aliases: set = {'issuer', 'receiver'}
    party_addr_aliases: set = {'issuer_addr', 'receiver_addr'}
    party_tax_aliases: set = {'issuer_tax', 'receiver_tax'}

    date_formats: tuple = (
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%d-%m-%Y',
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%m-%d-%Y',
        '%d.%m.%Y',
        '%d %b %Y',
        '%d %B %Y',
        '%b %d, %Y',
        '%B %d, %Y',
    )

    keyword_re: re.Pattern = re.compile(
        r'(invoice|inv\.?|bill\s*to|ship\s*to|buyer|seller|vendor|supplier|'
        r'date|due|tax|vat|gst|subtotal|sub\s*total|shipping|delivery|total)',
        flags=re.IGNORECASE,
    )

    score_weights: dict = {
        'semantic': 0.55,
        'lexical': 0.25,
        'value': 0.20,
    }

    kv_max_row_delta: int = 18

    address_tokens: tuple = ('street', 'st', 'road', 'rd', 'avenue', 'ave', 'lane', 'ln', 'blvd', 'city', 'zip')

    order_id_tokens: tuple = ('order', 'po', 'purchase order')
    due_positive_tokens: tuple = ('due',)
    due_negative_tokens: tuple = ('invoice date', 'issue')
    date_positive_tokens: tuple = ('invoice date', 'issue')
    date_negative_tokens: tuple = ('due',)

    issuer_negative_tokens: tuple = ('bill to', 'buyer', 'customer', 'ship to', 'receiver')
    receiver_negative_tokens: tuple = ('seller', 'vendor', 'supplier', 'from', 'issuer')


SUMMARY_CONFIG = SummaryExtractionConfig()

# AI Models
DEFAULT_GRANITE_MODEL_ID = 'ibm-granite/granite-embedding-small-english-r2'
