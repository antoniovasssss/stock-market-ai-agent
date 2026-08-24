import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------
# 1. Initialize OpenAI client
# ---------------------------------------

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise SystemExit("OPENAI_API_KEY is not set. Add it to your .env file.")

client = OpenAI(api_key=API_KEY)

# ----------------------------------------
# 2. Load datasets
# ----------------------------------------

nasdaq100_ca = pd.read_csv("nasdaq100_CA_practice.csv")

price_change = pd.read_csv(
    "nasdaq100_price_change_practice.csv"
)

# ----------------------------------------
# 3. Add YTD performance
# ----------------------------------------

nasdaq100_ca = nasdaq100_ca.merge(
    price_change[["symbol", "ytd"]],
    on="symbol",
    how="inner"
)


print("Combined DataFrame:")
print(nasdaq100_ca.head())

# ----------------------------------------
# 4. Classify companies into sectors
# ----------------------------------------

sectors = [
    "Technology",
    "Consumer Cyclical",
    "Industrials",
    "Utilities",
    "Healthcare",
    "Communication",
    "Energy",
    "Consumer Defensive",
    "Real Estate",
    "Financial"
]

nasdaq100_ca["sector"] = ""


for index, row in nasdaq100_ca.iterrows():

    prompt = f"""
Classify the following company into exactly ONE of these sectors:

Technology
Consumer Cyclical
Industrials
Utilities
Healthcare
Communication
Energy
Consumer Defensive
Real Estate
Financial

Company name: {row["name"]}
Stock symbol: {row["symbol"]}

Return ONLY the sector name.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    sector = response.choices[0].message.content.strip()

    nasdaq100_ca.loc[index, "sector"] = sector

# ----------------------------------------
# 5. Display sector counts
# ----------------------------------------

print("\nCompanies by sector:")

print(
    nasdaq100_ca["sector"].value_counts()
)


# ----------------------------------------
# 6. Ask OpenAI for recommendations
# ----------------------------------------

company_data = nasdaq100_ca[
    ["symbol", "name", "headQuarter", "ytd", "sector"]
].to_string(index=False)


prompt = f"""
Analyze the following NASDAQ-100 company data.

The data contains:
- Company name
- Stock symbol
- Headquarters
- YTD performance
- Sector

Identify the TWO best-performing sectors based primarily
on YTD performance.

Then recommend at least TWO companies from each of
the two best sectors.

For each recommended company, explain briefly why
it is a good candidate based on the supplied data.

Do not invent performance numbers.

Company data:

{company_data}
"""


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0
)

# ----------------------------------------
# 7. Store recommendations
# ----------------------------------------

stock_recommendations = response.choices[0].message.content

print("\nStock Recommendations:")
print(stock_recommendations)