"""
Feature engineering and characteristic generation package.
"""


def __getattr__(name):
    if name in ("CharacteristicBuilder", "run_feature_engineering", "FEATURE_COLUMNS"):
        from .builder import CharacteristicBuilder, FEATURE_COLUMNS, run_feature_engineering

        globals()["CharacteristicBuilder"] = CharacteristicBuilder
        globals()["FEATURE_COLUMNS"] = FEATURE_COLUMNS
        globals()["run_feature_engineering"] = run_feature_engineering
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["CharacteristicBuilder", "FEATURE_COLUMNS", "run_feature_engineering"]
