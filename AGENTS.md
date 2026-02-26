# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This repository is a collection of Python Dash applications for financial market data visualization, plus an Obsidian knowledge vault for investment research notes (Japanese).

### Applications

| App | Port | Directory | Description |
|-----|------|-----------|-------------|
| yield_curve_3d | 8051 | `yield_curve_3d/` | Main app: 3D yield curve & forward curve visualization (Japan/USA/UK/Euro/China/India/Gold/Silver) |
| stock_map_3D | 8050 | `stock_map_3D/` | 3D stock volume/price visualization (5-min candles) |
| stock_map_3D_1h | 8052 | `stock_map_3D_1h/` | 3D stock volume/price visualization (1-hour candles, 730 days) |
| forward_rate_chart_test | 8052 | `forward_rate_chart_test/` | Silver/Gold 1-month forward rate time-series chart |

**Note:** `stock_map_3D_1h` and `forward_rate_chart_test` both use port 8052 — run only one at a time.

### Running apps

Each app is started with `python app.py` from its directory, using the virtual environment at `/workspace/.venv`:

```bash
source /workspace/.venv/bin/activate
cd yield_curve_3d && python app.py     # port 8051
cd stock_map_3D && python app.py       # port 8050
cd forward_rate_chart_test && python app.py  # port 8052
```

### Key gotchas

- The `main` branch only has Obsidian vault + data files — no Python source code. The source code lives in feature branches (most complete: `fetch2-2-2`). If setting up from `main`, you must merge source from a feature branch first.
- `forward_rate_chart_test` reads data from `yield_curve_3d/data/` (cross-app dependency), not its own directory.
- README instructions reference Windows/PowerShell paths; on Linux, use standard Unix paths.
- No linting config, test framework, or CI/CD exists in this codebase. The `.gitignore` in `yield_curve_3d/` only ignores `api_keys.json`.
- Data fetching scripts (e.g., `fetch_usa_data.py`, `fetch_gold_data.py`) require network access and some require external API keys (see `api_keys.example.json`).
- The `stock_map_3D` and `stock_map_3D_1h` apps can fetch live stock data via yfinance, which requires network access.
