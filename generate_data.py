#!/usr/bin/env python3
"""
Data generator for GitHub Pages website.
Exports analysis results to JSON for the web frontend.
"""

import json
import os
from datetime import datetime
from economic_freedom_analysis import DataFetcher, CorrelationAnalyzer
from scipy import stats
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def generate_web_data(output_dir="docs"):
    """Generate JSON data files for the web frontend."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating data for web frontend")
    print("=" * 60)

    # Fetch all data
    fetcher = DataFetcher()

    economic_freedom = fetcher.fetch_heritage_economic_freedom()
    world_bank = fetcher.fetch_world_bank_data()
    corruption = fetcher.fetch_corruption_index()
    hdi = fetcher.fetch_human_development_index()
    happiness = fetcher.fetch_happiness_index()
    democracy = fetcher.fetch_democracy_index()
    social_mobility = fetcher.fetch_social_mobility_index()
    gini = fetcher.fetch_gini_index()
    purchasing_power = fetcher.fetch_purchasing_power_index()
    pollution = fetcher.fetch_pollution_index()
    crime = fetcher.fetch_crime_index()
    homicide = fetcher.fetch_unodc_homicide_rate()
    hale = fetcher.fetch_hale()
    mental_health = fetcher.fetch_mental_health_index()
    education = fetcher.fetch_education_index()

    # Merge datasets
    print("\nMerging datasets...")
    merged = economic_freedom.copy()

    datasets = [
        (world_bank, 'Country'),
        (corruption, 'Country'),
        (hdi, 'Country'),
        (happiness, 'Country'),
        (democracy, 'Country'),
        (social_mobility, 'Country'),
        (gini, 'Country'),
        (purchasing_power, 'Country'),
        (pollution, 'Country'),
        (crime, 'Country'),
        (homicide, 'Country'),
        (hale, 'Country'),
        (mental_health, 'Country'),
        (education, 'Country')
    ]

    for df, key in datasets:
        if df is not None and len(df) > 0:
            cols_to_use = [key] + [c for c in df.columns if c not in merged.columns]
            merged = merged.merge(df[cols_to_use], on=key, how='left')

    # Fill in missing Gini values from the dedicated Gini fetch
    if gini is not None and 'Gini_Index' in gini.columns:
        gini_dict = dict(zip(gini['Country'], gini['Gini_Index']))
        merged['Gini_Index'] = merged.apply(
            lambda row: gini_dict.get(row['Country'], row.get('Gini_Index'))
            if (row.get('Gini_Index') is None or (isinstance(row.get('Gini_Index'), float) and np.isnan(row.get('Gini_Index'))))
            else row.get('Gini_Index'),
            axis=1
        )

    print(f"  Merged dataset: {len(merged)} countries")

    # Calculate correlations
    analyzer = CorrelationAnalyzer(merged)
    results = analyzer.calculate_correlations()

    # Prepare data for JSON export
    output_data = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "num_countries": len(merged),
            "data_sources": [
                {"name": "Heritage Foundation Economic Freedom Index", "year": 2024},
                {"name": "World Bank Development Indicators", "year": 2022},
                {"name": "Transparency International CPI", "year": 2023},
                {"name": "UN Human Development Index", "year": 2022},
                {"name": "World Happiness Report", "year": 2024},
                {"name": "Economist Intelligence Unit Democracy Index", "year": 2023},
                {"name": "World Economic Forum Social Mobility Index", "year": 2020},
                {"name": "Numbeo Crime Index", "year": 2025},
                {"name": "UNODC Homicide Statistics (via World Bank)", "year": 2022},
                {"name": "WHO Healthy Life Expectancy (HALE)", "year": 2021},
                {"name": "WHO Mental Health Atlas", "year": 2021},
                {"name": "UNDP Education Index", "year": 2022}
            ]
        },
        "correlations": [],
        "countries": [],
        "scatter_data": {},
        "summary": {}
    }

    # Process correlations
    indices_config = {
        'GDP_Per_Capita': {'display_name': 'GDP per Capita', 'unit': 'USD', 'higher_better': True},
        'Life_Expectancy': {'display_name': 'Life Expectancy', 'unit': 'years', 'higher_better': True},
        'Infant_Mortality': {'display_name': 'Infant Mortality', 'unit': 'per 1000', 'higher_better': False},
        'Corruption_Score': {'display_name': 'Anti-Corruption Score', 'unit': '0-100', 'higher_better': True},
        'HDI': {'display_name': 'Human Development Index', 'unit': '0-1', 'higher_better': True},
        'Happiness_Score': {'display_name': 'Happiness Score', 'unit': '0-10', 'higher_better': True},
        'Social_Mobility_Score': {'display_name': 'Social Mobility Index', 'unit': '0-100', 'higher_better': True},
        'Purchasing_Power_Index': {'display_name': 'Purchasing Power', 'unit': '0-100', 'higher_better': True},
        'Pollution_Index': {'display_name': 'Pollution', 'unit': '0-100', 'higher_better': False},
        'Crime_Index': {'display_name': 'Crime (Numbeo)', 'unit': '0-100', 'higher_better': False},
        'Homicide_Rate': {'display_name': 'Homicide Rate (UNODC)', 'unit': 'per 100k', 'higher_better': False},
        'HALE': {'display_name': 'Healthy Life Expectancy (WHO)', 'unit': 'years', 'higher_better': True},
        'Mental_Health_Index': {'display_name': 'Mental Health Access (WHO)', 'unit': 'per 100k', 'higher_better': True},
        'Education_Index': {'display_name': 'Education Index (UNDP)', 'unit': '0-1', 'higher_better': True}
    }

    for col, result in results.items():
        config = indices_config.get(col, {})
        output_data["correlations"].append({
            "id": col,
            "name": result['display_name'],
            "correlation": round(result['correlation'], 4),
            "effective_correlation": round(result['effective_correlation'], 4),
            "p_value": result['p_value'],
            "significant": result['p_value'] < 0.05,
            "n": result['n'],
            "higher_better": config.get('higher_better', True),
            "unit": config.get('unit', '')
        })

    # Sort by effective correlation
    output_data["correlations"].sort(key=lambda x: x["effective_correlation"], reverse=True)

    # Process country data
    for _, row in merged.iterrows():
        country_data = {
            "name": row['Country'],
            "region": row.get('Region', 'Unknown'),
            "economic_freedom": row['Economic_Freedom_Score'] if not np.isnan(row['Economic_Freedom_Score']) else None
        }

        for col in indices_config.keys():
            if col in row:
                val = row[col]
                country_data[col.lower()] = round(val, 2) if not (isinstance(val, float) and np.isnan(val)) else None

        # Add Gini Index separately (for display, not correlation)
        if 'Gini_Index' in row:
            val = row['Gini_Index']
            country_data['gini_index'] = round(val, 1) if not (isinstance(val, float) and np.isnan(val)) else None

        # Add Democracy Score separately (for X-axis selection)
        if 'Democracy_Score' in row:
            val = row['Democracy_Score']
            country_data['democracy_score'] = round(val, 2) if not (isinstance(val, float) and np.isnan(val)) else None

        output_data["countries"].append(country_data)

    # Generate scatter plot data
    for col in indices_config.keys():
        if col in merged.columns:
            valid_data = merged[['Country', 'Economic_Freedom_Score', col]].dropna()
            scatter_points = []

            for _, row in valid_data.iterrows():
                scatter_points.append({
                    "country": row['Country'],
                    "x": round(row['Economic_Freedom_Score'], 2),
                    "y": round(row[col], 2) if col != 'HDI' else round(row[col], 3)
                })

            # Calculate trend line
            if len(valid_data) > 2:
                z = np.polyfit(valid_data['Economic_Freedom_Score'], valid_data[col], 1)
                x_min = valid_data['Economic_Freedom_Score'].min()
                x_max = valid_data['Economic_Freedom_Score'].max()

                output_data["scatter_data"][col] = {
                    "points": scatter_points,
                    "trend_line": {
                        "slope": round(z[0], 6),
                        "intercept": round(z[1], 4),
                        "x_range": [round(x_min, 2), round(x_max, 2)]
                    }
                }

    # Calculate summary statistics
    sig_positive = sum(1 for c in output_data["correlations"] if c["effective_correlation"] > 0 and c["significant"])
    sig_negative = sum(1 for c in output_data["correlations"] if c["effective_correlation"] < 0 and c["significant"])
    not_significant = sum(1 for c in output_data["correlations"] if not c["significant"])

    output_data["summary"] = {
        "positive_correlations": sig_positive,
        "negative_correlations": sig_negative,
        "not_significant": not_significant,
        "total_indices": len(output_data["correlations"]),
        "verdict": "supports" if sig_positive > sig_negative else ("contradicts" if sig_negative > sig_positive else "mixed")
    }

    # Calculate comparison: Economic Freedom vs Gini Index vs Democracy correlations
    # For Gini, lower is better (less inequality), so we invert the correlation for comparison
    comparison_indices = [
        ('GDP_Per_Capita', 'GDP per Capita', True),
        ('Life_Expectancy', 'Life Expectancy', True),
        ('Infant_Mortality', 'Infant Mortality', False),
        ('Corruption_Score', 'Anti-Corruption', True),
        ('HDI', 'Human Development Index', True),
        ('Happiness_Score', 'Happiness', True),
        ('Social_Mobility_Score', 'Social Mobility', True),
        ('Purchasing_Power_Index', 'Purchasing Power', True),
        ('Pollution_Index', 'Pollution', False),
        ('Crime_Index', 'Crime (Numbeo)', False),
        ('Homicide_Rate', 'Homicide Rate (UNODC)', False),
        ('HALE', 'Healthy Life Expectancy', True),
        ('Mental_Health_Index', 'Mental Health Access', True),
        ('Education_Index', 'Education Index', True)
    ]

    comparison_data = []
    ef_clear_wins = 0
    gini_clear_wins = 0
    democracy_clear_wins = 0
    ties = 0
    MARGIN_THRESHOLD = 0.05  # Minimum margin for a "clear win"

    for col, display_name, higher_better in comparison_indices:
        if col in merged.columns and 'Gini_Index' in merged.columns:
            # Economic Freedom correlation
            ef_valid = merged[['Economic_Freedom_Score', col]].dropna()
            if len(ef_valid) > 10:
                ef_corr, ef_p = stats.pearsonr(ef_valid['Economic_Freedom_Score'], ef_valid[col])
                ef_effective = ef_corr if higher_better else -ef_corr
            else:
                ef_corr, ef_p, ef_effective = None, None, None

            # Gini correlation (inverted since lower Gini = better)
            gini_valid = merged[['Gini_Index', col]].dropna()
            if len(gini_valid) > 10:
                gini_corr, gini_p = stats.pearsonr(gini_valid['Gini_Index'], gini_valid[col])
                # For Gini, negative correlation with good outcomes = good (less inequality = better outcomes)
                # So we negate it to make positive = good for comparison
                gini_effective = -gini_corr if higher_better else gini_corr
            else:
                gini_corr, gini_p, gini_effective = None, None, None

            # Democracy correlation
            democracy_valid = merged[['Democracy_Score', col]].dropna()
            if len(democracy_valid) > 10:
                democracy_corr, democracy_p = stats.pearsonr(democracy_valid['Democracy_Score'], democracy_valid[col])
                democracy_effective = democracy_corr if higher_better else -democracy_corr
            else:
                democracy_corr, democracy_p, democracy_effective = None, None, None

            # Determine winner (three-way comparison with margin threshold)
            winner = None
            effectives = {
                "economic_freedom": ef_effective,
                "equality": gini_effective,
                "democracy": democracy_effective
            }
            valid_effectives = {k: v for k, v in effectives.items() if v is not None}
            if len(valid_effectives) >= 2:
                sorted_items = sorted(valid_effectives.items(), key=lambda x: x[1], reverse=True)
                first_key, first_val = sorted_items[0]
                second_key, second_val = sorted_items[1]
                margin = first_val - second_val

                if margin >= MARGIN_THRESHOLD:
                    # Clear winner
                    winner = first_key
                    if first_key == "economic_freedom":
                        ef_clear_wins += 1
                    elif first_key == "equality":
                        gini_clear_wins += 1
                    elif first_key == "democracy":
                        democracy_clear_wins += 1
                else:
                    # Too close to call - it's a tie
                    winner = "tie"
                    ties += 1

            comparison_data.append({
                "metric": display_name,
                "metric_id": col,
                "economic_freedom": {
                    "correlation": round(ef_corr, 4) if ef_corr is not None else None,
                    "effective": round(ef_effective, 4) if ef_effective is not None else None,
                    "p_value": ef_p if ef_p is not None else None,
                    "significant": ef_p < 0.05 if ef_p is not None else False,
                    "n": len(ef_valid) if ef_corr is not None else 0
                },
                "equality": {
                    "correlation": round(gini_corr, 4) if gini_corr is not None else None,
                    "effective": round(gini_effective, 4) if gini_effective is not None else None,
                    "p_value": gini_p if gini_p is not None else None,
                    "significant": gini_p < 0.05 if gini_p is not None else False,
                    "n": len(gini_valid) if gini_corr is not None else 0
                },
                "democracy": {
                    "correlation": round(democracy_corr, 4) if democracy_corr is not None else None,
                    "effective": round(democracy_effective, 4) if democracy_effective is not None else None,
                    "p_value": democracy_p if democracy_p is not None else None,
                    "significant": democracy_p < 0.05 if democracy_p is not None else False,
                    "n": len(democracy_valid) if democracy_corr is not None else 0
                },
                "winner": winner
            })

    output_data["comparison"] = {
        "economic_freedom_clear_wins": ef_clear_wins,
        "equality_clear_wins": gini_clear_wins,
        "democracy_clear_wins": democracy_clear_wins,
        "ties": ties,
        "metrics": comparison_data
    }

    # Write JSON file
    output_file = os.path.join(output_dir, "data.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)

    print(f"\nData exported to: {output_file}")
    print(f"Summary: {sig_positive} positive, {sig_negative} negative, {not_significant} not significant")

    return output_data


if __name__ == "__main__":
    generate_web_data()
