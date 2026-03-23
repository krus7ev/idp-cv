from typing import Optional, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from .constants import CURRENCY_RE, FIRST_DIGIT_RE, NUMERIC_CLEAN_RE, SIGNED_INT_SPACED_RE, SPACE_RE, VALID_NUMBER_RE


class InvoiceSummary(BaseModel):
    invoice_number: Optional[str] = Field(
        default=None,
        alias='inv_no',
        title='Invoice Number',
        description='The unique identifier for the invoice, often found at the top of the document.',
        examples=['A123', '5414', 'INV-2024-001', 'INV-34566', '# 1234NA1'],
        json_schema_extra={
            'aliases': ['invoice:', 'invoice number:', 'inv no:', 'invoice no:', 'invoice #:'],
            'value_type': 'id',
        },
    )
    order_id: Optional[str] = Field(
        default=None,
        alias='order_id',
        title='Order ID',
        description='ID of the purchase oreder associated with the invoice',
        examples=['ES-2012-JH15820139-41012', '123545435154543', 'PO-34566'],
        json_schema_extra={
            'aliases': [
                'oid:',
                'order:',
                'order #:',
                'order id:',
                'order no:',
                'order num:',
                'po:',
                'po no:',
                'po num:',
                'po code:',
                'po number:',
            ],
            'value_type': 'id',
        },
    )
    invoice_date: Optional[str] = Field(
        default=None,
        alias='date',
        title='Invoice Date',
        description='The date when the invoice was issued.',
        examples=['2024-01-15', '02/20/2024', 'Mar 10, 2024'],
        json_schema_extra={
            'aliases': ['date:', 'invoice date:', 'date of issue:', 'issue date:', 'issued on:'],
            'value_type': 'date',
        },
    )
    due_date: Optional[str] = Field(
        default=None,
        alias='due',
        title='Due Date',
        description='The date by which the payment for the invoice is due.',
        examples=['2024-02-15', '03/20/2024', 'Apr 10, 2024'],
        json_schema_extra={
            'aliases': [
                'due date:',
                'due on:',
                'due by:',
                'payment due:',
                'payment by:',
                'pay by:',
                'payment date:',
                'pay date:',
            ],
            'value_type': 'date',
        },
    )
    total_amount: Optional[float] = Field(
        default=None,
        alias='total',
        title='Total Amount',
        description='The total amount due for the invoice, including all taxes and fees.',
        examples=[20123.45, 123.45, 123.45, 2123435.0, 12345.67],
        json_schema_extra={
            'aliases': [
                'total:',
                'balance:',
                'total balance:',
                'grand total:',
                'gross total:',
                'invoice total:',
                'total due:',
                'balance due:',
                'total amount:',
                'balance amount:',
                'full amount:',
            ],
            'value_type': 'amount',
        },
    )
    net_amount: Optional[float] = Field(
        default=None,
        alias='subtotal',
        title='Subtotal Amount',
        description='The subtotal amount before taxes and additional fees.',
        examples=[20123.45, 123.45, 123.45, 2123435.0, 12345.67],
        json_schema_extra={
            'aliases': ['net worth:', 'net amount:', 'net total:', 'subtotal:', 'sub total:', 'amount before tax:'],
            'value_type': 'amount',
        },
    )
    tax_amount: Optional[float] = Field(
        default=None,
        alias='tax_amount',
        title='Tax Amount',
        description='The total tax amount applied to the invoice.',
        examples=[2012.34, 12.34, 23.45, 123435.0, 2345.67],
        json_schema_extra={
            'aliases': [
                'tax:',
                'vat:',
                'gst:',
                'tax amount',
                'vat amount:',
                'gst amount:',
                'tax due:',
                'vat due:',
                'gst due:',
                'total tax:',
                'tax (12%):',
                'vat (12%):',
                'gst (12%):',
            ],
            'value_type': 'amount',
        },
    )
    tax_rate: Optional[float] = Field(
        default=None,
        alias='tax_rate',
        title='Tax Rate',
        description='The tax rate applied to the invoice - usually as a percentage (%).',
        examples=[10.0, 15.5, 0.07, 7.0, 0.15],
        json_schema_extra={
            'aliases': ['tax rate:', 'vat rate:', 'gst rate:', 'tax %', 'vat %', 'gst %'],
            'value_type': 'rate',
        },
    )
    shipping_cost: Optional[float] = Field(
        default=None,
        alias='shipping',
        title='Shipping Cost',
        description='The shipping cost for the invoice.',
        examples=[5.00, 5.00, 5.00, 5.00, 5.00],
        json_schema_extra={
            'aliases': ['shipping:', 'delivery:', 'freight:', 'shipping cost:', 'delivery cost:', 'freight cost:'],
            'value_type': 'amount',
        },
    )
    vendor_name: Optional[str] = Field(
        default=None,
        alias='issuer',
        title='Vendor Name',
        description='The name of the vendor or supplier issuing the invoice.',
        examples=['John Doe', 'Acme Corp', 'Global Supplies Inc.', 'Tech Solutions LLC'],
        json_schema_extra={'aliases': ['seller:', 'vendor:', 'supplier:', 'from:', 'issuer:'], 'value_type': 'name'},
    )
    buyer_name: Optional[str] = Field(
        default=None,
        alias='receiver',
        title='Buyer Name',
        description='The name of the buyer or customer that the invoice is billed to.',
        examples=['Jane Smith', 'XYZ Company', 'Innovatech Ltd.', 'Creative Agency Inc.'],
        json_schema_extra={
            'aliases': [
                'bill to:',
                'buyer:',
                'customer:',
                'receiver:',
                'recepient:',
                'client:',
                'issued to:',
            ],
            'value_type': 'name',
        },
    )
    vendor_tax_id: Optional[str] = Field(
        default=None,
        alias='issuer_tax',
        title='Vendor Tax ID',
        description='The tax identification number of the vendor or supplier issuing the invoice.',
        examples=['534340453', '12-56765', '123-98-45334', 'TAXID564456', 'TX 76543210', 'TX-987654'],
        json_schema_extra={
            'aliases': ['tax id:', 'seller tax id:', 'vendor tax id:', 'supplier tax id:'],
            'value_type': 'id',
        },
    )
    vendor_banking: Optional[str] = Field(
        default=None,
        alias='issuer_banking',
        title='Vendor Banking Details',
        description='The bank acccount details of the issuer at which payment is due.',
        examples=['GB13LKIT26418644296553', 'GB30ICA043176037560423'],
        json_schema_extra={
            'aliases': [
                'banking:',
                'bank account:',
                'account number:',
                'account no:',
                'acc. no:',
                'iban:',
            ],
            'value_type': 'id',
        },
    )
    buyer_tax_id: Optional[str] = Field(
        default=None,
        alias='receiver_tax',
        title='Buyer Tax ID',
        description='The tax identification number of the buyer or customer that the invoice is billed to.',
        examples=['534340453', '12-56765', '123-98-45334', 'TAXID564456', 'TX 76543210', 'TX-987654'],
        json_schema_extra={
            'aliases': ['tax id:', 'buyer tax id:', 'customer tax id:', 'client tax id:', 'billing tax id:'],
            'value_type': 'id',
        },
    )

    vendor_address: Optional[str] = Field(
        default=None,
        alias='issuer_addr',
        title='Vendor Address',
        description='The address of the vendor or supplier issuing the invoice.',
        examples=['123 Main St, Anytown, USA', '456 Elm St, Othercity, USA', '789 Oak St, Sometown, USA'],
        json_schema_extra={
            'aliases': ['address:', 'issuer address:', 'seller address:', 'vendor address:', 'from address:'],
            'value_type': 'address',
        },
    )
    buyer_address: Optional[str] = Field(
        default=None,
        alias='receiver_addr',
        title='Buyer Address',
        description='The address of the buyer or customer that the invoice is billed to.',
        examples=['123 Main St, Anytown, USA', '456 Elm St, Othercity, USA', '789 Oak St, Sometown, USA'],
        json_schema_extra={
            'aliases': [
                'address:',
                'bill to address:',
                'buyer address:',
                'client address:',
                'customer address:',
                'receiver address:',
                'billing address:',
                'shipping address:',
                'ship to:',
                'ship to address:',
                'delivery address:',
                'deliver to:',
                'destination:',
                'destination address:',
                'addr:',
            ],
            'value_type': 'address',
        },
    )


class ValidatedTableLine(BaseModel):
    @staticmethod
    def _has_negative_sign(text: str) -> bool:
        first_digit = FIRST_DIGIT_RE.search(text)
        if not first_digit:
            return False

        before_number = text[: first_digit.start()]
        if '-' in before_number:
            return True

        return False

    @field_validator('tax_rate', mode='before', check_fields=False)
    @classmethod
    def parse_tax_rate(cls, v: Union[str, float, int], info: Optional[ValidationInfo] = None) -> Optional[float]:
        """Normalize tax rate values, handling"""
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if not isinstance(v, str):
            print(f'Expected number or "str" input to parse as tax rate, got "{type(v).__name__}"!')
            return

        text = v.strip().strip('% ')
        return cls._extract_numeric_value(text)

    @field_validator(
        'quantity',
        'unit_price',
        'total_amount',
        'net_amount',
        'tax_amount',
        'shipping_cost',
        mode='before',
        check_fields=False,
    )
    @classmethod
    def parse_numeric(
        cls, v: Union[str, float, int], info: Optional[ValidationInfo] = None
    ) -> Optional[Union[int, float]]:
        """Normalize quantity/amount values with a quantity-only integer fast-path."""
        field_name = info.field_name if info is not None else None
        is_monetary = field_name == 'unit_price' or field_name == 'total_amount'

        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
        if not isinstance(v, str):
            print(f'Expected number or "str" input to parse as numeric, got "{type(v).__name__}"!')
            return

        # Quantity-only fast-path before currency normalization
        if not is_monetary and SIGNED_INT_SPACED_RE.fullmatch(v):
            return int(SPACE_RE.sub('', v))

        text = CURRENCY_RE.sub('', v).strip()
        return cls._extract_numeric_value(text)

    @classmethod
    def _extract_numeric_value(cls, text: str) -> float:
        numeric = NUMERIC_CLEAN_RE.sub('', text)
        numeric = SPACE_RE.sub('', numeric)

        if not numeric:
            return None

        normalized = cls._normalize_separators(numeric)
        match = VALID_NUMBER_RE.search(normalized)
        value = float(match.group()) if match else 0.0
        return -value if cls._has_negative_sign(text) and value != 0.0 else value

    @staticmethod
    def _normalize_separators(num: str) -> str:
        if ',' not in num and '.' not in num:
            return num

        if ',' in num and '.' in num:
            decimal_sep = ',' if num.rfind(',') > num.rfind('.') else '.'
            thousand_sep = '.' if decimal_sep == ',' else ','

            return num.replace(thousand_sep, '').replace(decimal_sep, '.')

        sep = ',' if ',' in num else '.'
        parts = num.split(sep)

        all_thousands = (
            len(parts) > 1 and all(part.isdigit() for part in parts) and all(len(part) == 3 for part in parts[1:])
        )

        if all_thousands:
            return ''.join(parts)

        int_part = ''.join(parts[:-1])
        frac_part = parts[-1]
        return f'{int_part}.{frac_part}' if frac_part else int_part


class TableLine(ValidatedTableLine):
    description: str = Field(
        default=None,
        alias='desc',
        title='Description',
        description='Name or brief description of the good or service being invoiced.',
        examples=['Widget A', 'Consulting Services', 'Software License', 'Office Supplies'],
        json_schema_extra={
            'aliases': ['description', 'desc', 'item description', 'service', 'product', 'details', 'item'],
            'value_type': 'string',
        },
    )
    quantity: int = Field(
        default=None,
        alias='qty',
        title='Quantity',
        description='The quantity of the good or service being invoiced.',
        examples=[1, 9, 10, 112],
        json_schema_extra={'aliases': ['quantity', 'qty', 'quant', 'units', 'count'], 'value_type': 'amount'},
    )
    unit_price: float = Field(
        default=None,
        alias='price',
        title='Unit Price',
        description='The unit price amount of the good or service being invoiced.',
        examples=[20.0, 15.5, 0.07, 7.0, 0.15],
        json_schema_extra={
            'aliases': ['unit price', 'price', 'rate', 'unit cost', 'net price', 'cost per unit', 'cpu'],
            'value_type': 'amount',
        },
    )
    total_amount: Optional[float] = Field(
        default=None,
        alias='total',
        title='Total Amount',
        description='The total-amount column of an invoice table line.',
        examples=[20123.45, 123.45, 123.45, 2123435.0, 12345.67],
        json_schema_extra={
            'aliases': [
                'gross_worth',
                'gross total',
                'grand total',
                'invoice total',
                'total_worth',
                'total amount',
                'total due',
                'balance due',
                'total balance',
                'total',
                'balance',
                'balance amount',
                'full amount',
            ],
            'value_type': 'amount',
        },
    )
    net_amount: Optional[float] = Field(
        default=None,
        alias='net',
        title='Net Amount',
        description='The net amount before taxes and additional fees.',
        examples=[20123.45, 123.45, 123.45, 2123435.0, 12345.67],
        json_schema_extra={
            'aliases': ['net worth', 'net amount', 'net total', 'subtotal', 'sub total', 'amount before tax'],
            'value_type': 'amount',
        },
    )
    tax_amount: Optional[float] = Field(
        default=None,
        alias='tax',
        title='Tax Amount',
        description='The total tax amount applied to the invoice.',
        examples=[2012.34, 12.34, 23.45, 123435.0, 2345.67],
        json_schema_extra={
            'aliases': [
                'tax',
                'vat',
                'gst',
                'tax amount',
                'vat amount',
                'gst amount',
                'tax due',
                'vat due',
                'gst due',
                'total tax',
            ],
            'value_type': 'amount',
        },
    )
    tax_rate: Optional[float] = Field(
        default=None,
        alias='tax_rate',
        title='Tax Rate',
        description='The tax rate applied to the invoice - usually as a percentage (%).',
        examples=[10.0, 15.5, 0.07, 7.0, 0.15],
        json_schema_extra={
            'aliases': ['tax (%)', 'vat (%)', 'gst (%)', 'tax %', 'vat %', 'gst %', 'tax rate', 'vat rate', 'gst rate'],
            'value_type': 'rate',
        },
    )
    shipping_cost: Optional[float] = Field(
        default=None,
        alias='shipping',
        title='Shipping Cost',
        description='The shipping cost for the invoice.',
        examples=[5.00, 5.00, 5.00, 5.00, 5.00],
        json_schema_extra={
            'aliases': ['shipping', 'delivery', 'freight', 'shipping cost', 'delivery cost', 'freight cost'],
            'value_type': 'amount',
        },
    )
