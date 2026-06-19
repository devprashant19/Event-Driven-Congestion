# Fix Dashboard Stability, Syntactic Issues, and Dataset Pipeline Hardcoding

## Description
This PR addresses several critical stability and syntactical bugs identified across the project. It removes brittle hardcoded paths to improve reproducibility and prevents Streamlit from crashing unexpectedly when navigating between tabs.

## Changes Included
1. **`dashboard.py`**:
   - Fixed `NameError` related to missing `json` module when directly accessing the "Post-Event Analysis" tab. 
   - Refactored `st.info` f-strings to avoid using inner double quotes that caused `SyntaxError` in Python < 3.12.
   - Removed `importlib.reload(recommendation_engine)` at the top level to avoid performance penalties and forced recompilations during Streamlit production interaction.
   - Modified checkpoint search to use an environment variable `MODEL_CHECKPOINT_DIR` with a fallback, avoiding crashes if the training config name or seed changes.

2. **`prepare_astram_dataset.py`**:
   - Replaced the hardcoded dataset CSV path with an `--event_csv` argument using `argparse`. The script can now be flexibly executed on different environments.

3. **`recommendation_engine.py`**:
   - Swapped out silent failures and empty variable assignments when `data/city_network.json` is missing. It now raises a clear `FileNotFoundError` to prevent obscure divide-by-zero or missing topology errors later during path calculations.

## Testing
- Verified `dashboard.py` syntax parsing for older python environments.
- Confirmed `prepare_astram_dataset.py` argparse works with the new flag.
