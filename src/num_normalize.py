import re
from decimal import Decimal, InvalidOperation

def normalize_number(s: str):
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"[^\d.,%-]", "", s)
    is_pct = "%" in s
    s = s.replace("%", "")
    # fix european style
    if re.match(r"^\d{1,3}(\.\d{3})+,\d+$", s):
        s = s.replace(".", "").replace(",", ".")
    elif re.match(r"^\d{1,3}(,\d{3})+\.\d+$", s):
        s = s.replace(",", "")
    try:
        val = float(Decimal(s))
        return val / 100.0 if is_pct else val
    except (InvalidOperation, ValueError):
        return None
