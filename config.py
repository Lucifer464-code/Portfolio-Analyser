# ==========================================================
# GLOBAL CONFIGURATION
# ==========================================================

# Core assumptions
BASE_RISK_FREE_RATE = 0.04
TRADING_DAYS = 365.25

# Optimization parameters
REGULARIZATION_STRENGTH = 0.05
MONTE_CARLO_PORTFOLIOS = 10000

# Risk profile constraints
RISK_PROFILES = {
    "Conservative": {
        "max_weight": 0.20,
        "min_weight": 0.01,
        "rf_multiplier": 1.2
    },
    "Moderate": {
        "max_weight": 0.30,
        "min_weight": 0.02,
        "rf_multiplier": 1.0
    },
    "Aggressive": {
        "max_weight": 0.50,
        "min_weight": 0.02,
        "rf_multiplier": 0.8
    }
}

# Default transaction cost (flat %)
DEFAULT_TRANSACTION_COST = 0.001  # 0.10%

# Factor ETF proxies
FACTOR_ETFS = {
    "Market": "SPY",
    "Size": "IWM",
    "Momentum": "MTUM",
    "Value": "VLUE"
}
