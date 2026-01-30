# 🪙 Gold Brain Automation

A Python-based automation that tracks gold prices and saves them to a Supabase database.

## 🚀 How it Works
- **Automation:** Powered by GitHub Actions.
- **Trigger:** Runs automatically every 15 minutes.
- **Data Source:** Fetches live gold prices using `yfinance`.
- **Database:** Stores data in a Supabase table named `gold_prices`.

## 🛠️ Setup Requirements
To run this yourself, you need to set up the following **GitHub Secrets**:
1. `SUPABASE_URL`: Your Supabase project URL.
2. `SUPABASE_KEY`: Your Supabase API key.

## 📂 Project Structure
- `.github/workflows/main.yml`: The automation rules.
- `brain_logic.py`: The Python logic for fetching and saving data.
