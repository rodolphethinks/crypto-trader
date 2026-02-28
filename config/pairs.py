"""
Trading pair configurations loaded from NOFEE-LIST.md.
"""
from pathlib import Path
from typing import List, Dict

_NOFEE_LIST_PATH = Path(__file__).resolve().parent.parent / "NOFEE-LIST.md"


def load_nofee_pairs() -> List[str]:
    """Load all 0-fee trading pairs from NOFEE-LIST.md."""
    pairs = []
    if _NOFEE_LIST_PATH.exists():
        with open(_NOFEE_LIST_PATH, "r") as f:
            for line in f:
                pair = line.strip()
                if pair and not pair.startswith("#"):
                    pairs.append(pair)
    return pairs


def get_major_pairs() -> List[str]:
    """High-liquidity pairs ideal for backtesting."""
    return [
        "BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "BNBUSDC",
        "ADAUSDC", "DOGEUSDC", "AVAXUSDC", "DOTUSDC", "LINKUSDC",
        "USDCUSDT",
    ]


def get_stablecoin_pairs() -> List[str]:
    """Stablecoin pairs for range/mean-reversion strategies."""
    return ["USDCUSDT", "USDCEUR", "EURUSDC"]


def get_altcoin_pairs() -> List[str]:
    """Mid-cap altcoins suitable for momentum/breakout strategies."""
    return [
        "SUIUSDC", "APTUSDC", "INJUSDC", "TIAUSDC", "SEIUSDC",
        "NEARUSDC", "ATOMUSDC", "RENDERUSDC", "FETUSDC", "ARBUSDC",
        "OPUSDC", "PEPEUSDC", "BONKUSDC", "WIFUSDC", "KASUSDC",
    ]


def get_meme_pairs() -> List[str]:
    """Meme coins — high volatility, suitable for scalping/momentum."""
    return [
        "DOGEUSDC", "SHIBUSDC", "PEPEUSDC", "BONKUSDC", "FLOKIUSDC",
        "WIFUSDC", "TRUMPUSDC", "BABYDOGEUSDC", "FARTCOINUSDC",
    ]


def get_pair_categories() -> Dict[str, List[str]]:
    """Return pairs grouped by category."""
    return {
        "majors": get_major_pairs(),
        "stablecoins": get_stablecoin_pairs(),
        "altcoins": get_altcoin_pairs(),
        "memes": get_meme_pairs(),
    }


# Pre-loaded lists
ALL_NOFEE_PAIRS = load_nofee_pairs()
MAJOR_PAIRS = get_major_pairs()
STABLECOIN_PAIRS = get_stablecoin_pairs()
