"""
Data ingestion and data processing package.
"""


def __getattr__(name):
    if name in ("StockDataIngestion", "run_ingestion"):
        from .ingestion import StockDataIngestion, run_ingestion
        globals()["StockDataIngestion"] = StockDataIngestion
        globals()["run_ingestion"] = run_ingestion
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["StockDataIngestion", "run_ingestion"]
