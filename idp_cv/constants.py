import re

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
ADDR_NER_TAGS = {'LOC', 'FAC', 'CARDINAL', 'GPE', 'ORG', 'PERSON', 'DATE', 'PRODUCT'}
LOC_NER_TAGS = {'LOC', 'FAC', 'GPE'}
NAME_NER_TAGS = {'PERSON', 'NORP', 'ORG', 'GPE'}

# AI Models
DEFAULT_GRANITE_MODEL_ID = 'ibm-granite/granite-embedding-small-english-r2'


C_SWAP_FILEDS = [
    ('issuer', 'receiver'),
    ('issuer_addr', 'receiver_addr'),
    ('issuer_tax', 'receiver_tax'),
]
