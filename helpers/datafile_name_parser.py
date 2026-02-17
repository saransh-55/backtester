from pathlib import Path

def to_datafile_name(symbol: str, timeframe: str, exchange_id: str, ext: str = "csv") -> str:
    """
    Build a filename from symbol, timeframe, exchange_id, and extension.

    Args:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        timeframe: Timeframe string (e.g., "1m", "1h")
        exchange_id: Exchange identifier (e.g., "binance")
        ext: File extension without dot (e.g., "csv", "parquet")

    Returns:
        Formatted filename: symbol_timeframe_exchange_id.ext
    """
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    return f"{safe_symbol}_{timeframe}_{exchange_id}.{ext}"


def from_datafile_name(filename: str) -> tuple[str, str, str, str]:
    """
    Parse a filename into its components: symbol, timeframe, exchange_id, and extension.

    Args:
        filename: Filename to parse (e.g., "BTC-USDT_1m_binance.csv")

    Returns:
        Tuple of (symbol, timeframe, exchange_id, ext)
        Symbol will have "/" restored (e.g., "BTC/USDT")

    Raises:
        ValueError: If filename doesn't match expected format
    """
    # Remove path if present
    filename = Path(filename).name

    # Split extension
    parts = filename.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Filename must have an extension: {filename}")

    base_name, ext = parts

    # Split base name into symbol_timeframe_exchange
    components = base_name.split("_")
    if len(components) < 3:
        raise ValueError(f"Filename must follow format 'symbol_timeframe_exchange.ext': {filename}")

    # Last two parts are timeframe and exchange_id
    exchange_id = components[-1]
    timeframe = components[-2]
    # Everything else is the symbol (rejoin with underscore in case symbol had underscores)
    safe_symbol = "_".join(components[:-2])

    # Restore "/" in symbol (assuming BTC-USDT format)
    symbol = safe_symbol.replace("-", "/", 1)  # Only replace first dash

    return symbol, timeframe, exchange_id, ext
