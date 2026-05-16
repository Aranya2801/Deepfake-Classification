#!/usr/bin/env python3
"""
Quick-run script for Zomato Review Analysis pipeline.
Usage:  python run_analysis.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.sentiment_engine import run_full_pipeline
from src.pipeline.visualizer import generate_all_charts

def main():
    print("=" * 65)
    print("  🍽️  ZOMATO REVIEW ANALYSIS — PIPELINE RUNNER")
    print("=" * 65)

    enriched_df, report = run_full_pipeline(
        input_path  = "data/raw/zomato_reviews_raw.csv",
        output_path = "data/processed/zomato_reviews_enriched.csv",
        train_ml    = True,
    )

    print(f"\n{'─'*65}")
    print(f"  📊 ANALYSIS REPORT")
    print(f"{'─'*65}")
    print(f"  Total Reviews   : {report.total_reviews:,}")
    print(f"  ✅ Positive      : {report.positive_count:,}  ({report.positive_count/report.total_reviews:.1%})")
    print(f"  ❌ Negative      : {report.negative_count:,}  ({report.negative_count/report.total_reviews:.1%})")
    print(f"  ⚪ Neutral       : {report.neutral_count:,}  ({report.neutral_count/report.total_reviews:.1%})")
    print(f"  ⭐ Avg Rating    : {report.avg_rating}")
    print(f"  📡 Avg VADER     : {report.avg_compound_score}")
    print(f"\n  Top Positive Keywords:")
    for word, count in report.top_positive_keywords[:5]:
        print(f"    • {word:<20} ({count} mentions)")
    print(f"\n  Top Negative Keywords:")
    for word, count in report.top_negative_keywords[:5]:
        print(f"    • {word:<20} ({count} mentions)")
    print(f"\n  Top Restaurants by Sentiment:")
    for i, (rest, score) in enumerate(list(report.restaurant_scores.items())[:5], 1):
        print(f"    {i}. {rest:<30} score={score:+.3f}")

    print(f"\n{'─'*65}")
    print("  📈 Generating visualizations...")
    saved = generate_all_charts(enriched_df, output_dir="assets/")
    print(f"  ✅ {len(saved)} charts saved to assets/")
    print(f"{'─'*65}")
    print("\n  🚀 Launch dashboard:  streamlit run dashboard/app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
