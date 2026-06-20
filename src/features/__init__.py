# ======================================================================
# AURELINE LABS — FEATURE FACTORY
# Every feature module follows the same contract:
#   Input:  DataFrame with OHLCV columns
#   Output: Same DataFrame with new feature columns added
#   Rules:  No look-ahead bias. No raw prices. Only relative values.
# ======================================================================
from src.features.momentum   import add_momentum_features
from src.features.volatility import add_volatility_features
from src.features.trend      import add_trend_features
from src.features.volume     import add_volume_features
from src.features.regime     import add_regime_features


def build_all_features(data):
    """
    Runs the complete feature factory pipeline.
    Applies all feature modules in sequence.
    Returns a DataFrame with all engineered features added.
    """
    data = add_momentum_features(data)
    data = add_volatility_features(data)
    data = add_trend_features(data)
    data = add_volume_features(data)
    data = add_regime_features(data)
    return data


def all_feature_cols():
    """
    Returns the complete list of feature column names
    produced by the factory. Single source of truth.
    """
    from src.features.momentum   import MOMENTUM_FEATURES
    from src.features.volatility import VOLATILITY_FEATURES
    from src.features.trend      import TREND_FEATURES
    from src.features.volume     import VOLUME_FEATURES
    from src.features.regime     import REGIME_FEATURES
    return (MOMENTUM_FEATURES + VOLATILITY_FEATURES +
            TREND_FEATURES + VOLUME_FEATURES + REGIME_FEATURES)