import pandas as pd
from openai import AzureOpenAI
import json
import re
import os
from dotenv import load_dotenv

# ==============================
# 🔧 CONFIGURATION
# ==============================

load_dotenv()

endpoint = os.getenv("ENDPOINT")
deployment = os.getenv("DEPLOYMENT_NAME")
api_version = os.getenv("API_VERSION")
subscription_key = os.getenv("OPENAI_API_KEY")

input_file = "C:/Users/manish_jha/Desktop/2025 Hackathon/backend/all_store_reviews_labelled_mapped_store5.csv"
output_detailed = "C:/Users/manish_jha/Desktop/2025 Hackathon/backend/store_5_reviews_theme_weighted_scores.csv"
output_summary = "C:/Users/manish_jha/Desktop/2025 Hackathon/backend/store_5_period_theme_average_scores.csv"

REVIEW_COLUMN = "text"
STORE_COLUMN = "store_name"       # 👈 must exist in your CSV
PERIOD_COLUMN = "time_period"     # 👈 must exist in your CSV (e.g. “< 1 month”, “1–6 months”, etc.)

# ==============================
# 🔗 CONNECT TO AZURE OPENAI
# ==============================
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# ==============================
# 🧩 HELPER FUNCTIONS
# ==============================
def safe_json_parse(text):
    """Extract and safely parse JSON from LLM output."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            clean_text = match.group(0).replace("\n", "").replace("\r", "").replace("\t", "")
            return json.loads(clean_text)
        else:
            return {"overall_sentiment_score": 3, "themes": []}
    except Exception as e:
        print(f"⚠️ JSON parse error: {e}")
        return {"overall_sentiment_score": 3, "themes": []} 


def analyze_review(review):
    """Analyze sentiment + themes using Azure OpenAI."""
    if pd.isna(review) or not str(review).strip():
        return {"overall_sentiment_score": 3, "themes": []}

    prompt = f"""
You are a retail analytics assistant.
Return ONLY valid JSON — no markdown, no explanations.

Analyze the following customer review and respond ONLY in valid JSON with:
- "overall_sentiment_score": integer 1–5 where:
    1 = Very Negative
    2 = Negative
    3 = Neutral
    4 = Positive
    5 = Very Positive
- "themes": a list of objects, each containing:
    - "name": one of [Waiting Time, Staff Behaviour, Cleanliness, Ease of locating items, Other]
    - "sentiment_score": integer 1–5
    - "confidence": a number between 0 and 100

Example output:
{{
  "overall_sentiment_score": 2,
  "themes": [
    {{"name": "Staff Behaviour", "sentiment_score": 1, "confidence": 95}},
    {{"name": "Waiting Time", "sentiment_score": 2, "confidence": 85}}
  ]
}}

Review: "{review}"
"""

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        result = safe_json_parse(text)
        return result

    except Exception as e:
        print(f"Error on review: {e}")
        return {"overall_sentiment_score": 3, "themes": []}


def get_theme(themes, index, field="name"):
    """Get theme info by confidence order."""
    try:
        sorted_themes = sorted(themes, key=lambda t: t.get("confidence", 0), reverse=True)
        return sorted_themes[index][field] if len(sorted_themes) > index else ""
    except Exception:
        return ""


def multiply_weight(score, weight):
    """Calculate weighted sentiment score for a theme."""
    try:
        return round(float(score) * weight, 2) if score not in [None, "", "Error"] else ""
    except Exception:
        return ""


# ==============================
# 🚀 STEP 1: ANALYZE REVIEWS
# ==============================
df = pd.read_csv(input_file)
print(f"🔍 Loaded {len(df)} reviews...")

results = df[REVIEW_COLUMN].apply(analyze_review)

# Extract structured data
df["overall_sentiment_score"] = results.apply(lambda x: x.get("overall_sentiment_score"))
df["primary_theme"] = results.apply(lambda x: get_theme(x.get("themes", []), 0, "name"))
df["primary_theme_sentiment_score"] = results.apply(lambda x: get_theme(x.get("themes", []), 0, "sentiment_score"))
df["secondary_theme"] = results.apply(lambda x: get_theme(x.get("themes", []), 1, "name"))
df["secondary_theme_sentiment_score"] = results.apply(lambda x: get_theme(x.get("themes", []), 1, "sentiment_score"))
df["tertiary_theme"] = results.apply(lambda x: get_theme(x.get("themes", []), 2, "name"))
df["tertiary_theme_sentiment_score"] = results.apply(lambda x: get_theme(x.get("themes", []), 2, "sentiment_score"))

# Weighted scores
df["primary_theme_weighted_score"] = df["primary_theme_sentiment_score"].apply(lambda x: multiply_weight(x, 0.5))
df["secondary_theme_weighted_score"] = df["secondary_theme_sentiment_score"].apply(lambda x: multiply_weight(x, 0.3))
df["tertiary_theme_weighted_score"] = df["tertiary_theme_sentiment_score"].apply(lambda x: multiply_weight(x, 0.2))

# Save detailed file
df.to_csv(output_detailed, index=False, encoding="utf-8-sig")
print(f"✅ Step 1 complete: Detailed review analysis saved to\n{output_detailed}")

# ==============================
# 📊 STEP 2: THEME AVERAGE AGGREGATION (BY STORE + PERIOD + THEME)
# ==============================
theme_data = pd.DataFrame()

for level in ["primary", "secondary", "tertiary"]:
    theme_col = f"{level}_theme"
    score_col = f"{level}_theme_weighted_score"

    temp = df[[STORE_COLUMN, PERIOD_COLUMN, theme_col, score_col]].rename(
        columns={theme_col: "theme", score_col: "weighted_score"}
    )
    theme_data = pd.concat([theme_data, temp], ignore_index=True)

# Clean data
theme_data = theme_data.dropna(subset=["theme", "weighted_score"])
theme_data = theme_data[theme_data["theme"] != ""]
theme_data["weighted_score"] = pd.to_numeric(theme_data["weighted_score"], errors="coerce")

# Group by store, period, theme
theme_summary = (
    theme_data.groupby([STORE_COLUMN, PERIOD_COLUMN, "theme"], as_index=False)
    .agg(avg_weighted_score=("weighted_score", "mean"), count=("weighted_score", "count"))
    .sort_values(by=[STORE_COLUMN, PERIOD_COLUMN, "avg_weighted_score"], ascending=[True, True, False])
)
theme_summary["avg_weighted_score"] = theme_summary["avg_weighted_score"].round(2)

# Save summary file
theme_summary.to_csv(output_summary, index=False, encoding="utf-8-sig")
print(f"✅ Step 2 complete: Theme averages by store & period saved to\n{output_summary}")

# Optional preview
print("\n📊 Store + Period + Theme Summary:")
print(theme_summary.head())
