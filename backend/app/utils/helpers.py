def format_currency(value: float, symbol: str = "₹") -> str:
    return f"{symbol}{value:,.2f}"


