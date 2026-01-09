#!/usr/bin/env python3
"""
Economic Freedom vs Quality of Life Indices Analysis
=====================================================
This script fetches economic freedom data and correlates it with various
quality of life indices to examine relationships between economic freedom
and societal outcomes.

Data Sources:
- Heritage Foundation Economic Freedom Index
- World Bank (GDP per capita, Life Expectancy)
- Transparency International (Corruption Perceptions Index)
- UN Human Development Index
- World Happiness Report
- Global Peace Index
"""

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class DataFetcher:
    """Fetches data from various sources for economic and quality of life indices."""

    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_heritage_economic_freedom(self):
        """
        Fetch Heritage Foundation Economic Freedom Index data.
        Uses their publicly available data.
        """
        print("Fetching Heritage Economic Freedom Index...")

        # Heritage provides data in a downloadable format
        # We'll use a recent dataset - they publish annually
        url = "https://www.heritage.org/index/pages/all-country-scores"

        try:
            # Try to fetch from Heritage API/data endpoint
            # Note: Heritage doesn't have a public API, so we use backup data approach
            response = requests.get(
                "https://www.heritage.org/index/sites/default/files/2024_IndexofEconomicFreedom.xlsx",
                timeout=30
            )
            if response.status_code == 200:
                import io
                df = pd.read_excel(io.BytesIO(response.content), sheet_name=0)
                df.to_csv(f"{self.cache_dir}/heritage_economic_freedom.csv", index=False)
                return df
        except Exception as e:
            print(f"  Could not fetch live Heritage data: {e}")

        # Fallback: Use embedded 2024 data for key countries
        print("  Using embedded Heritage Foundation data (2024)...")
        data = self._get_heritage_embedded_data()
        df = pd.DataFrame(data)
        df.to_csv(f"{self.cache_dir}/heritage_economic_freedom.csv", index=False)
        return df

    def _get_heritage_embedded_data(self):
        """Embedded Heritage Economic Freedom Index 2024 data for major countries."""
        return {
            'Country': [
                'Singapore', 'Switzerland', 'Ireland', 'Taiwan', 'New Zealand',
                'Estonia', 'Luxembourg', 'Netherlands', 'Denmark', 'Sweden',
                'Germany', 'Australia', 'United Kingdom', 'Finland', 'Canada',
                'United States', 'Norway', 'South Korea', 'Japan', 'Austria',
                'Israel', 'Chile', 'France', 'Spain', 'Portugal',
                'Italy', 'Poland', 'Czech Republic', 'Slovenia', 'Slovakia',
                'Hungary', 'Greece', 'Mexico', 'Turkey', 'Colombia',
                'South Africa', 'Brazil', 'India', 'Indonesia', 'Philippines',
                'Thailand', 'Vietnam', 'China', 'Russia', 'Egypt',
                'Nigeria', 'Pakistan', 'Bangladesh', 'Argentina', 'Venezuela',
                'Cuba', 'North Korea', 'Iran', 'Zimbabwe', 'Ukraine',
                'Malaysia', 'Costa Rica', 'Uruguay', 'Panama', 'Jamaica',
                'Peru', 'Dominican Republic', 'Morocco', 'Kenya', 'Ghana',
                'Rwanda', 'Botswana', 'Mauritius', 'Saudi Arabia', 'UAE',
                'Qatar', 'Kuwait', 'Bahrain', 'Jordan', 'Tunisia'
            ],
            'Economic_Freedom_Score': [
                83.5, 83.2, 82.0, 80.7, 78.9,
                78.6, 78.4, 77.8, 77.6, 77.5,
                73.7, 73.5, 72.7, 72.6, 72.0,
                70.1, 68.3, 72.6, 69.3, 71.2,
                69.0, 70.1, 65.7, 65.5, 67.2,
                63.8, 69.0, 71.9, 66.5, 67.4,
                64.8, 57.8, 63.7, 55.2, 58.1,
                55.7, 53.4, 52.9, 58.5, 56.3,
                62.8, 53.7, 48.2, 52.3, 49.7,
                54.4, 49.4, 49.8, 50.3, 24.8,
                24.1, 2.9, 42.5, 39.5, 54.1,
                67.3, 65.8, 68.9, 66.4, 62.4,
                62.7, 61.5, 59.3, 53.6, 56.4,
                63.2, 64.9, 72.1, 57.5, 70.2,
                63.5, 62.7, 66.9, 57.4, 54.8
            ],
            'Region': [
                'Asia-Pacific', 'Europe', 'Europe', 'Asia-Pacific', 'Asia-Pacific',
                'Europe', 'Europe', 'Europe', 'Europe', 'Europe',
                'Europe', 'Asia-Pacific', 'Europe', 'Europe', 'Americas',
                'Americas', 'Europe', 'Asia-Pacific', 'Asia-Pacific', 'Europe',
                'Middle East', 'Americas', 'Europe', 'Europe', 'Europe',
                'Europe', 'Europe', 'Europe', 'Europe', 'Europe',
                'Europe', 'Europe', 'Americas', 'Europe', 'Americas',
                'Sub-Saharan Africa', 'Americas', 'Asia-Pacific', 'Asia-Pacific', 'Asia-Pacific',
                'Asia-Pacific', 'Asia-Pacific', 'Asia-Pacific', 'Europe', 'Middle East',
                'Sub-Saharan Africa', 'Asia-Pacific', 'Asia-Pacific', 'Americas', 'Americas',
                'Americas', 'Asia-Pacific', 'Middle East', 'Sub-Saharan Africa', 'Europe',
                'Asia-Pacific', 'Americas', 'Americas', 'Americas', 'Americas',
                'Americas', 'Americas', 'Middle East', 'Sub-Saharan Africa', 'Sub-Saharan Africa',
                'Sub-Saharan Africa', 'Sub-Saharan Africa', 'Sub-Saharan Africa', 'Middle East', 'Middle East',
                'Middle East', 'Middle East', 'Middle East', 'Middle East', 'Middle East'
            ]
        }

    def fetch_world_bank_data(self):
        """Fetch World Bank indicators: GDP per capita, Life Expectancy, etc."""
        print("Fetching World Bank data...")

        indicators = {
            'NY.GDP.PCAP.CD': 'GDP_Per_Capita',
            'SP.DYN.LE00.IN': 'Life_Expectancy',
            'SE.ADT.LITR.ZS': 'Literacy_Rate',
            'SP.DYN.IMRT.IN': 'Infant_Mortality',
            'SL.UEM.TOTL.ZS': 'Unemployment_Rate'
        }

        all_data = []

        for indicator_code, indicator_name in indicators.items():
            try:
                url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}?format=json&date=2022&per_page=300"
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    if len(data) > 1 and data[1]:
                        for item in data[1]:
                            if item['value'] is not None:
                                all_data.append({
                                    'Country': item['country']['value'],
                                    'Country_Code': item['countryiso3code'],
                                    indicator_name: item['value']
                                })
            except Exception as e:
                print(f"  Error fetching {indicator_name}: {e}")

        if all_data:
            df = pd.DataFrame(all_data)
            # Aggregate by country (taking the latest value)
            df = df.groupby(['Country', 'Country_Code']).first().reset_index()
            df.to_csv(f"{self.cache_dir}/world_bank_data.csv", index=False)
            return df

        # Fallback embedded data
        print("  Using embedded World Bank data...")
        return pd.DataFrame(self._get_world_bank_embedded_data())

    def _get_world_bank_embedded_data(self):
        """Embedded World Bank data for key countries."""
        return {
            'Country': [
                'Singapore', 'Switzerland', 'Ireland', 'Taiwan', 'New Zealand',
                'Estonia', 'Luxembourg', 'Netherlands', 'Denmark', 'Sweden',
                'Germany', 'Australia', 'United Kingdom', 'Finland', 'Canada',
                'United States', 'Norway', 'South Korea', 'Japan', 'Austria',
                'Israel', 'Chile', 'France', 'Spain', 'Portugal',
                'Italy', 'Poland', 'Czech Republic', 'Slovenia', 'Slovakia',
                'Hungary', 'Greece', 'Mexico', 'Turkey', 'Colombia',
                'South Africa', 'Brazil', 'India', 'Indonesia', 'Philippines',
                'Thailand', 'Vietnam', 'China', 'Russia', 'Egypt',
                'Nigeria', 'Pakistan', 'Bangladesh', 'Argentina', 'Venezuela',
                'Cuba', 'North Korea', 'Iran', 'Zimbabwe', 'Ukraine',
                'Malaysia', 'Costa Rica', 'Uruguay', 'Panama', 'Jamaica',
                'Peru', 'Dominican Republic', 'Morocco', 'Kenya', 'Ghana',
                'Rwanda', 'Botswana', 'Mauritius', 'Saudi Arabia', 'UAE',
                'Qatar', 'Kuwait', 'Bahrain', 'Jordan', 'Tunisia'
            ],
            'GDP_Per_Capita': [
                65233, 93457, 103685, 32756, 48249,
                28247, 126426, 57768, 67803, 56424,
                51204, 65366, 46125, 53983, 54966,
                76330, 106149, 32255, 33815, 53268,
                54771, 16265, 44408, 30103, 24521,
                34776, 18688, 27221, 28439, 21088,
                18728, 20867, 11497, 10674, 6630,
                6776, 8918, 2389, 4788, 3499,
                7066, 4164, 12720, 15345, 3699,
                2184, 1596, 2688, 13650, None,
                None, None, 4078, 1773, 4534,
                12364, 13612, 20795, 17357, 5859,
                7126, 10111, 3795, 2099, 2363,
                966, 7347, 10216, 30436, 49451,
                83891, 38124, 29103, 4543, 3893
            ],
            'Life_Expectancy': [
                84.1, 84.0, 82.8, 81.0, 82.5,
                78.8, 82.6, 82.0, 81.5, 83.2,
                81.2, 84.5, 81.8, 82.2, 82.7,
                77.5, 83.2, 83.7, 84.8, 82.0,
                83.0, 80.7, 82.5, 83.3, 82.2,
                83.5, 78.0, 79.4, 81.3, 77.8,
                76.7, 80.8, 75.1, 78.0, 77.3,
                64.9, 75.3, 70.4, 71.8, 71.7,
                78.7, 75.4, 78.2, 72.4, 71.5,
                53.9, 67.3, 72.4, 76.7, 72.1,
                79.0, 72.0, 76.7, 61.5, 71.6,
                76.3, 80.3, 78.4, 78.5, 75.2,
                77.0, 74.1, 76.7, 67.0, 64.1,
                69.6, 61.1, 74.4, 78.2, 78.4,
                80.2, 78.7, 78.0, 74.5, 76.7
            ],
            'Infant_Mortality': [
                1.5, 3.4, 2.6, 3.8, 3.5,
                2.1, 3.2, 3.1, 2.9, 2.0,
                3.0, 2.9, 3.5, 1.8, 4.2,
                5.4, 1.6, 2.5, 1.7, 2.8,
                2.7, 5.7, 3.3, 2.5, 2.4,
                2.4, 3.5, 2.3, 1.6, 4.4,
                3.6, 3.1, 11.1, 8.2, 11.7,
                26.5, 12.4, 26.6, 18.8, 20.8,
                7.0, 15.6, 5.1, 4.4, 16.3,
                71.2, 52.3, 23.3, 8.0, 21.0,
                4.0, 14.0, 11.7, 34.5, 6.4,
                6.2, 6.9, 5.6, 12.3, 11.6,
                10.7, 23.5, 17.1, 29.8, 32.1,
                27.7, 27.1, 11.3, 5.0, 5.0,
                5.4, 6.3, 5.3, 12.4, 12.8
            ]
        }

    def fetch_corruption_index(self):
        """Fetch Transparency International Corruption Perceptions Index."""
        print("Fetching Corruption Perceptions Index...")

        # Embedded CPI data (2023)
        data = {
            'Country': [
                'Singapore', 'Switzerland', 'Ireland', 'Taiwan', 'New Zealand',
                'Estonia', 'Luxembourg', 'Netherlands', 'Denmark', 'Sweden',
                'Germany', 'Australia', 'United Kingdom', 'Finland', 'Canada',
                'United States', 'Norway', 'South Korea', 'Japan', 'Austria',
                'Israel', 'Chile', 'France', 'Spain', 'Portugal',
                'Italy', 'Poland', 'Czech Republic', 'Slovenia', 'Slovakia',
                'Hungary', 'Greece', 'Mexico', 'Turkey', 'Colombia',
                'South Africa', 'Brazil', 'India', 'Indonesia', 'Philippines',
                'Thailand', 'Vietnam', 'China', 'Russia', 'Egypt',
                'Nigeria', 'Pakistan', 'Bangladesh', 'Argentina', 'Venezuela',
                'Cuba', 'North Korea', 'Iran', 'Zimbabwe', 'Ukraine',
                'Malaysia', 'Costa Rica', 'Uruguay', 'Panama', 'Jamaica',
                'Peru', 'Dominican Republic', 'Morocco', 'Kenya', 'Ghana',
                'Rwanda', 'Botswana', 'Mauritius', 'Saudi Arabia', 'UAE',
                'Qatar', 'Kuwait', 'Bahrain', 'Jordan', 'Tunisia'
            ],
            'Corruption_Score': [
                83, 82, 77, 68, 85,
                76, 78, 79, 90, 82,
                78, 75, 71, 87, 76,
                69, 84, 63, 73, 71,
                62, 66, 71, 60, 61,
                56, 54, 57, 56, 54,
                42, 49, 31, 34, 40,
                41, 36, 39, 34, 33,
                35, 41, 42, 26, 30,
                25, 29, 24, 37, 13,
                42, 17, 24, 24, 36,
                50, 55, 74, 36, 44,
                36, 32, 38, 31, 43,
                53, 55, 50, 52, 68,
                58, 46, 42, 46, 40
            ]
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.cache_dir}/corruption_index.csv", index=False)
        return df

    def fetch_human_development_index(self):
        """Fetch UN Human Development Index data."""
        print("Fetching Human Development Index...")

        # HDI 2022 data
        data = {
            'Country': [
                'Singapore', 'Switzerland', 'Ireland', 'Taiwan', 'New Zealand',
                'Estonia', 'Luxembourg', 'Netherlands', 'Denmark', 'Sweden',
                'Germany', 'Australia', 'United Kingdom', 'Finland', 'Canada',
                'United States', 'Norway', 'South Korea', 'Japan', 'Austria',
                'Israel', 'Chile', 'France', 'Spain', 'Portugal',
                'Italy', 'Poland', 'Czech Republic', 'Slovenia', 'Slovakia',
                'Hungary', 'Greece', 'Mexico', 'Turkey', 'Colombia',
                'South Africa', 'Brazil', 'India', 'Indonesia', 'Philippines',
                'Thailand', 'Vietnam', 'China', 'Russia', 'Egypt',
                'Nigeria', 'Pakistan', 'Bangladesh', 'Argentina', 'Venezuela',
                'Cuba', 'North Korea', 'Iran', 'Zimbabwe', 'Ukraine',
                'Malaysia', 'Costa Rica', 'Uruguay', 'Panama', 'Jamaica',
                'Peru', 'Dominican Republic', 'Morocco', 'Kenya', 'Ghana',
                'Rwanda', 'Botswana', 'Mauritius', 'Saudi Arabia', 'UAE',
                'Qatar', 'Kuwait', 'Bahrain', 'Jordan', 'Tunisia'
            ],
            'HDI': [
                0.939, 0.967, 0.950, 0.926, 0.939,
                0.899, 0.927, 0.946, 0.952, 0.947,
                0.950, 0.946, 0.940, 0.942, 0.935,
                0.927, 0.966, 0.929, 0.920, 0.926,
                0.915, 0.860, 0.910, 0.911, 0.874,
                0.906, 0.881, 0.895, 0.926, 0.855,
                0.851, 0.893, 0.781, 0.855, 0.758,
                0.717, 0.760, 0.644, 0.713, 0.710,
                0.803, 0.726, 0.788, 0.821, 0.728,
                0.548, 0.544, 0.670, 0.849, 0.699,
                0.764, 0.733, 0.780, 0.550, 0.773,
                0.807, 0.806, 0.830, 0.820, 0.709,
                0.762, 0.766, 0.698, 0.601, 0.602,
                0.548, 0.708, 0.796, 0.875, 0.937,
                0.875, 0.831, 0.888, 0.736, 0.732
            ]
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.cache_dir}/hdi.csv", index=False)
        return df

    def fetch_happiness_index(self):
        """Fetch World Happiness Report data."""
        print("Fetching World Happiness Report data...")

        # World Happiness Report 2024 data
        data = {
            'Country': [
                'Singapore', 'Switzerland', 'Ireland', 'Taiwan', 'New Zealand',
                'Estonia', 'Luxembourg', 'Netherlands', 'Denmark', 'Sweden',
                'Germany', 'Australia', 'United Kingdom', 'Finland', 'Canada',
                'United States', 'Norway', 'South Korea', 'Japan', 'Austria',
                'Israel', 'Chile', 'France', 'Spain', 'Portugal',
                'Italy', 'Poland', 'Czech Republic', 'Slovenia', 'Slovakia',
                'Hungary', 'Greece', 'Mexico', 'Turkey', 'Colombia',
                'South Africa', 'Brazil', 'India', 'Indonesia', 'Philippines',
                'Thailand', 'Vietnam', 'China', 'Russia', 'Egypt',
                'Nigeria', 'Pakistan', 'Bangladesh', 'Argentina', 'Venezuela',
                'Cuba', 'North Korea', 'Iran', 'Zimbabwe', 'Ukraine',
                'Malaysia', 'Costa Rica', 'Uruguay', 'Panama', 'Jamaica',
                'Peru', 'Dominican Republic', 'Morocco', 'Kenya', 'Ghana',
                'Rwanda', 'Botswana', 'Mauritius', 'Saudi Arabia', 'UAE',
                'Qatar', 'Kuwait', 'Bahrain', 'Jordan', 'Tunisia'
            ],
            'Happiness_Score': [
                6.587, 7.060, 6.838, 6.535, 7.029,
                5.972, 7.122, 7.319, 7.583, 7.344,
                6.892, 7.057, 6.749, 7.804, 6.900,
                6.725, 7.302, 5.882, 6.129, 7.097,
                7.473, 6.172, 6.687, 6.421, 6.224,
                6.324, 6.260, 6.822, 6.650, 6.469,
                6.041, 5.931, 6.678, 4.614, 5.881,
                5.275, 6.330, 4.036, 5.450, 5.961,
                6.368, 5.485, 5.973, 5.466, 4.288,
                4.740, 4.657, 4.359, 6.024, 5.607,
                5.353, None, 4.876, 3.204, 4.872,
                5.975, 6.609, 6.494, 6.265, 5.890,
                5.526, 5.569, 5.184, 4.509, 4.605,
                3.728, 4.941, 5.902, 6.594, 6.765,
                6.596, 6.170, 6.173, 5.328, 4.596
            ]
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.cache_dir}/happiness.csv", index=False)
        return df

    def fetch_peace_index(self):
        """Fetch Global Peace Index data."""
        print("Fetching Global Peace Index data...")

        # Global Peace Index 2024 (inverted so higher = more peaceful)
        data = {
            'Country': [
                'Singapore', 'Switzerland', 'Ireland', 'Taiwan', 'New Zealand',
                'Estonia', 'Luxembourg', 'Netherlands', 'Denmark', 'Sweden',
                'Germany', 'Australia', 'United Kingdom', 'Finland', 'Canada',
                'United States', 'Norway', 'South Korea', 'Japan', 'Austria',
                'Israel', 'Chile', 'France', 'Spain', 'Portugal',
                'Italy', 'Poland', 'Czech Republic', 'Slovenia', 'Slovakia',
                'Hungary', 'Greece', 'Mexico', 'Turkey', 'Colombia',
                'South Africa', 'Brazil', 'India', 'Indonesia', 'Philippines',
                'Thailand', 'Vietnam', 'China', 'Russia', 'Egypt',
                'Nigeria', 'Pakistan', 'Bangladesh', 'Argentina', 'Venezuela',
                'Cuba', 'North Korea', 'Iran', 'Zimbabwe', 'Ukraine',
                'Malaysia', 'Costa Rica', 'Uruguay', 'Panama', 'Jamaica',
                'Peru', 'Dominican Republic', 'Morocco', 'Kenya', 'Ghana',
                'Rwanda', 'Botswana', 'Mauritius', 'Saudi Arabia', 'UAE',
                'Qatar', 'Kuwait', 'Bahrain', 'Jordan', 'Tunisia'
            ],
            'Peace_Score': [  # Higher = more peaceful (inverted from GPI raw scores)
                81.5, 83.2, 82.7, 72.4, 84.1,
                78.5, 82.3, 79.2, 82.8, 78.3,
                77.4, 79.6, 74.3, 82.5, 79.4,
                61.7, 82.9, 71.2, 83.4, 80.1,
                39.2, 72.8, 73.6, 74.8, 82.1,
                77.9, 76.4, 80.2, 81.3, 78.7,
                77.1, 74.5, 48.3, 56.2, 52.4,
                51.8, 53.6, 56.4, 69.3, 58.4,
                67.3, 72.1, 68.4, 36.5, 57.8,
                42.1, 42.8, 62.4, 69.1, 48.5,
                65.3, 38.4, 49.3, 54.2, 28.6,
                72.8, 79.5, 78.4, 69.5, 66.4,
                61.3, 67.8, 72.5, 56.3, 62.1,
                58.4, 73.2, 80.5, 54.6, 75.4,
                79.2, 71.3, 76.8, 65.4, 68.2
            ]
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.cache_dir}/peace_index.csv", index=False)
        return df

    def fetch_democracy_index(self):
        """Fetch Democracy Index data (Economist Intelligence Unit)."""
        print("Fetching Democracy Index data...")

        # Democracy Index 2023
        data = {
            'Country': [
                'Singapore', 'Switzerland', 'Ireland', 'Taiwan', 'New Zealand',
                'Estonia', 'Luxembourg', 'Netherlands', 'Denmark', 'Sweden',
                'Germany', 'Australia', 'United Kingdom', 'Finland', 'Canada',
                'United States', 'Norway', 'South Korea', 'Japan', 'Austria',
                'Israel', 'Chile', 'France', 'Spain', 'Portugal',
                'Italy', 'Poland', 'Czech Republic', 'Slovenia', 'Slovakia',
                'Hungary', 'Greece', 'Mexico', 'Turkey', 'Colombia',
                'South Africa', 'Brazil', 'India', 'Indonesia', 'Philippines',
                'Thailand', 'Vietnam', 'China', 'Russia', 'Egypt',
                'Nigeria', 'Pakistan', 'Bangladesh', 'Argentina', 'Venezuela',
                'Cuba', 'North Korea', 'Iran', 'Zimbabwe', 'Ukraine',
                'Malaysia', 'Costa Rica', 'Uruguay', 'Panama', 'Jamaica',
                'Peru', 'Dominican Republic', 'Morocco', 'Kenya', 'Ghana',
                'Rwanda', 'Botswana', 'Mauritius', 'Saudi Arabia', 'UAE',
                'Qatar', 'Kuwait', 'Bahrain', 'Jordan', 'Tunisia'
            ],
            'Democracy_Score': [
                6.22, 9.14, 9.19, 8.99, 9.61,
                7.96, 8.81, 9.00, 9.28, 9.39,
                8.80, 8.96, 8.54, 9.30, 8.88,
                7.85, 9.81, 8.09, 8.33, 8.41,
                7.93, 7.98, 7.99, 8.07, 8.03,
                7.68, 6.80, 7.97, 7.96, 7.17,
                6.50, 7.45, 5.57, 4.35, 6.80,
                7.05, 6.78, 7.18, 6.71, 6.12,
                6.04, 2.60, 1.94, 2.22, 2.93,
                4.12, 3.25, 5.50, 6.62, 2.36,
                2.46, 1.08, 1.96, 2.60, 5.42,
                7.30, 8.29, 8.66, 7.24, 7.11,
                6.60, 6.15, 5.04, 5.05, 5.56,
                3.10, 7.81, 8.14, 2.08, 2.90,
                3.78, 3.83, 3.38, 3.93, 5.51
            ]
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.cache_dir}/democracy_index.csv", index=False)
        return df

    def fetch_social_mobility_index(self):
        """Fetch Global Social Mobility Index data (World Economic Forum)."""
        print("Fetching Social Mobility Index data...")

        # Global Social Mobility Index 2020 (WEF)
        # Score 0-100, higher = more social mobility
        data = {
            'Country': [
                'Singapore', 'Switzerland', 'Ireland', 'Taiwan', 'New Zealand',
                'Estonia', 'Luxembourg', 'Netherlands', 'Denmark', 'Sweden',
                'Germany', 'Australia', 'United Kingdom', 'Finland', 'Canada',
                'United States', 'Norway', 'South Korea', 'Japan', 'Austria',
                'Israel', 'Chile', 'France', 'Spain', 'Portugal',
                'Italy', 'Poland', 'Czech Republic', 'Slovenia', 'Slovakia',
                'Hungary', 'Greece', 'Mexico', 'Turkey', 'Colombia',
                'South Africa', 'Brazil', 'India', 'Indonesia', 'Philippines',
                'Thailand', 'Vietnam', 'China', 'Russia', 'Egypt',
                'Nigeria', 'Pakistan', 'Bangladesh', 'Argentina', 'Venezuela',
                'Cuba', 'North Korea', 'Iran', 'Zimbabwe', 'Ukraine',
                'Malaysia', 'Costa Rica', 'Uruguay', 'Panama', 'Jamaica',
                'Peru', 'Dominican Republic', 'Morocco', 'Kenya', 'Ghana',
                'Rwanda', 'Botswana', 'Mauritius', 'Saudi Arabia', 'UAE',
                'Qatar', 'Kuwait', 'Bahrain', 'Jordan', 'Tunisia'
            ],
            'Social_Mobility_Score': [
                74.6, 82.4, 75.9, None, 77.7,
                73.3, 77.3, 82.4, 85.2, 83.5,
                78.8, 75.1, 74.4, 83.6, 76.1,
                70.4, 83.2, 67.4, 76.1, 78.0,
                68.7, 59.1, 76.7, 70.0, 72.0,
                67.4, 65.1, 72.2, 75.6, 66.7,
                63.2, 65.7, 52.6, 52.5, 50.4,
                41.4, 52.1, 42.7, 49.2, 49.7,
                55.5, 52.0, 61.5, 64.4, 47.9,
                37.3, 36.1, 38.9, 57.1, 42.0,
                None, None, 48.1, 38.2, 56.4,
                60.1, 64.1, 66.8, 56.8, 55.1,
                53.6, 52.0, 48.4, 40.8, 44.9,
                35.8, 51.0, 59.8, 52.0, 58.3,
                54.2, 56.4, 59.3, 50.5, 52.1
            ]
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.cache_dir}/social_mobility_index.csv", index=False)
        return df

    def fetch_purchasing_power_index(self):
        """Fetch Purchasing Power Index data from Numbeo 2024."""
        print("Fetching Purchasing Power Index data...")

        # Real data from Numbeo Purchasing Power Index 2024
        # Source: https://www.numbeo.com/quality-of-life/rankings_by_country.jsp?title=2024
        # Higher = more purchasing power
        data = {
            'Country': [
                'Singapore', 'Switzerland', 'Ireland', 'Taiwan', 'New Zealand',
                'Estonia', 'Luxembourg', 'Netherlands', 'Denmark', 'Sweden',
                'Germany', 'Australia', 'United Kingdom', 'Finland', 'Canada',
                'United States', 'Norway', 'South Korea', 'Japan', 'Austria',
                'Israel', 'Chile', 'France', 'Spain', 'Portugal',
                'Italy', 'Poland', 'Czech Republic', 'Slovenia', 'Slovakia',
                'Hungary', 'Greece', 'Mexico', 'Turkey', 'Colombia',
                'South Africa', 'Brazil', 'India', 'Indonesia', 'Philippines',
                'Thailand', 'Vietnam', 'China', 'Russia', 'Egypt',
                'Nigeria', 'Pakistan', 'Bangladesh', 'Argentina', 'Venezuela',
                'Cuba', 'North Korea', 'Iran', 'Zimbabwe', 'Ukraine',
                'Malaysia', 'Costa Rica', 'Uruguay', 'Panama', 'Jamaica',
                'Peru', 'Dominican Republic', 'Morocco', 'Kenya', 'Ghana',
                'Rwanda', 'Botswana', 'Mauritius', 'Saudi Arabia', 'UAE',
                'Qatar', 'Kuwait', 'Bahrain', 'Jordan', 'Tunisia'
            ],
            'Purchasing_Power_Index': [
                102.99, 118.91, 81.99, 80.03, 93.01,  # Singapore, Switzerland, Ireland, Taiwan, New Zealand
                69.05, 148.92, 103.25, 103.32, 94.56,  # Estonia, Luxembourg, Netherlands, Denmark, Sweden
                101.05, 93.60, 90.06, 97.26, 83.26,  # Germany, Australia, UK, Finland, Canada
                120.91, 94.95, 90.57, 99.53, 84.43,  # USA, Norway, South Korea, Japan, Austria
                79.51, 36.39, 83.45, 78.18, 46.51,  # Israel, Chile, France, Spain, Portugal
                62.80, 66.19, 64.91, 67.97, 55.77,  # Italy, Poland, Czech Republic, Slovenia, Slovakia
                49.85, 42.07, 37.96, 39.32, 28.07,  # Hungary, Greece, Mexico, Turkey, Colombia
                84.66, 30.08, 60.69, 25.76, 25.86,  # South Africa, Brazil, India, Indonesia, Philippines
                32.94, 32.49, 60.54, 41.47, 15.23,  # Thailand, Vietnam, China, Russia, Egypt
                9.38, 24.45, 25.81, 35.79, 12.61,  # Nigeria, Pakistan, Bangladesh, Argentina, Venezuela
                None, None, 21.16, None, 32.23,  # Cuba, North Korea, Iran, Zimbabwe, Ukraine
                62.14, None, 47.01, 34.65, None,  # Malaysia, Costa Rica, Uruguay, Panama, Jamaica
                27.30, None, 30.37, 30.61, None,  # Peru, Dominican Republic, Morocco, Kenya, Ghana
                None, None, None, 105.11, 98.70,  # Rwanda, Botswana, Mauritius, Saudi Arabia, UAE
                127.40, 128.52, None, 36.37, None  # Qatar, Kuwait, Bahrain, Jordan, Tunisia
            ]
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.cache_dir}/purchasing_power_index.csv", index=False)
        return df

    def fetch_military_strength_index(self):
        """Fetch Military Strength Index data from Global Firepower 2025."""
        print("Fetching Military Strength Index data...")

        # Real data from Global Firepower 2025
        # Source: https://www.globalfirepower.com/countries-listing.php
        # PowerIndex score - LOWER = STRONGER military
        # Converting to 0-100 scale where HIGHER = STRONGER for consistency
        # Formula: 100 * (1 - PowerIndex/4) capped at 0-100
        raw_scores = {
            'Singapore': 0.5271, 'Switzerland': 0.7869, 'Ireland': 2.1103, 'Taiwan': 0.3988, 'New Zealand': 1.9039,
            'Estonia': 2.2917, 'Luxembourg': 2.6415, 'Netherlands': 0.6412, 'Denmark': 0.8109, 'Sweden': 0.4835,
            'Germany': 0.2601, 'Australia': 0.3298, 'United Kingdom': 0.1785, 'Finland': 0.8437, 'Canada': 0.5179,
            'United States': 0.0744, 'Norway': 0.6811, 'South Korea': 0.1656, 'Japan': 0.1839, 'Austria': 1.3704,
            'Israel': 0.2661, 'Chile': 0.8361, 'France': 0.1878, 'Spain': 0.3242, 'Portugal': 0.6856,
            'Italy': 0.2164, 'Poland': 0.3776, 'Czech Republic': 0.9994, 'Slovenia': 2.1016, 'Slovakia': 1.3978,
            'Hungary': 1.0259, 'Greece': 0.5337, 'Mexico': 0.5965, 'Turkey': 0.1902, 'Colombia': 0.8353,
            'South Africa': 0.6889, 'Brazil': 0.2415, 'India': 0.1184, 'Indonesia': 0.2557, 'Philippines': 0.6987,
            'Thailand': 0.4536, 'Vietnam': 0.4024, 'China': 0.0788, 'Russia': 0.0788, 'Egypt': 0.3427,
            'Nigeria': 0.5771, 'Pakistan': 0.2513, 'Bangladesh': 0.6062, 'Argentina': 0.6013, 'Venezuela': 0.8882,
            'Cuba': 1.3286, 'North Korea': 0.6016, 'Iran': 0.3048, 'Zimbabwe': 2.3863, 'Ukraine': 0.3755,
            'Malaysia': 0.7429, 'Peru': 0.8588, 'Morocco': 1.1273, 'Kenya': 1.8135,
            'Saudi Arabia': 0.4201, 'UAE': 1.0186, 'Qatar': 1.4307, 'Kuwait': 1.6982, 'Bahrain': 1.7448,
            'Jordan': 1.6139, 'Tunisia': 1.9538
        }

        # Convert to 0-100 scale (higher = stronger)
        data = {
            'Country': [],
            'Military_Strength_Index': []
        }

        countries = [
            'Singapore', 'Switzerland', 'Ireland', 'Taiwan', 'New Zealand',
            'Estonia', 'Luxembourg', 'Netherlands', 'Denmark', 'Sweden',
            'Germany', 'Australia', 'United Kingdom', 'Finland', 'Canada',
            'United States', 'Norway', 'South Korea', 'Japan', 'Austria',
            'Israel', 'Chile', 'France', 'Spain', 'Portugal',
            'Italy', 'Poland', 'Czech Republic', 'Slovenia', 'Slovakia',
            'Hungary', 'Greece', 'Mexico', 'Turkey', 'Colombia',
            'South Africa', 'Brazil', 'India', 'Indonesia', 'Philippines',
            'Thailand', 'Vietnam', 'China', 'Russia', 'Egypt',
            'Nigeria', 'Pakistan', 'Bangladesh', 'Argentina', 'Venezuela',
            'Cuba', 'North Korea', 'Iran', 'Zimbabwe', 'Ukraine',
            'Malaysia', 'Costa Rica', 'Uruguay', 'Panama', 'Jamaica',
            'Peru', 'Dominican Republic', 'Morocco', 'Kenya', 'Ghana',
            'Rwanda', 'Botswana', 'Mauritius', 'Saudi Arabia', 'UAE',
            'Qatar', 'Kuwait', 'Bahrain', 'Jordan', 'Tunisia'
        ]

        for country in countries:
            data['Country'].append(country)
            if country in raw_scores:
                # Convert: lower PowerIndex = stronger, so invert
                # Score of 0.07 (USA) -> ~98, Score of 2.5 -> ~37
                converted = max(0, min(100, 100 * (1 - raw_scores[country] / 3)))
                data['Military_Strength_Index'].append(round(converted, 1))
            else:
                data['Military_Strength_Index'].append(None)

        df = pd.DataFrame(data)
        df.to_csv(f"{self.cache_dir}/military_strength_index.csv", index=False)
        return df

    def fetch_gini_index(self):
        """Fetch Gini Index data (World Bank) - measure of income inequality."""
        print("Fetching Gini Index data...")

        # Gini Index data (World Bank, most recent available year per country)
        # Scale 0-100, higher = more inequality (0 = perfect equality, 100 = perfect inequality)
        data = {
            'Country': [
                'Singapore', 'Switzerland', 'Ireland', 'Taiwan', 'New Zealand',
                'Estonia', 'Luxembourg', 'Netherlands', 'Denmark', 'Sweden',
                'Germany', 'Australia', 'United Kingdom', 'Finland', 'Canada',
                'United States', 'Norway', 'South Korea', 'Japan', 'Austria',
                'Israel', 'Chile', 'France', 'Spain', 'Portugal',
                'Italy', 'Poland', 'Czech Republic', 'Slovenia', 'Slovakia',
                'Hungary', 'Greece', 'Mexico', 'Turkey', 'Colombia',
                'South Africa', 'Brazil', 'India', 'Indonesia', 'Philippines',
                'Thailand', 'Vietnam', 'China', 'Russia', 'Egypt',
                'Nigeria', 'Pakistan', 'Bangladesh', 'Argentina', 'Venezuela',
                'Cuba', 'North Korea', 'Iran', 'Zimbabwe', 'Ukraine',
                'Malaysia', 'Costa Rica', 'Uruguay', 'Panama', 'Jamaica',
                'Peru', 'Dominican Republic', 'Morocco', 'Kenya', 'Ghana',
                'Rwanda', 'Botswana', 'Mauritius', 'Saudi Arabia', 'UAE',
                'Qatar', 'Kuwait', 'Bahrain', 'Jordan', 'Tunisia'
            ],
            'Gini_Index': [
                45.9, 33.1, 30.6, 33.6, 32.0,
                30.6, 32.3, 28.4, 28.2, 28.8,
                31.7, 34.3, 35.1, 27.3, 33.3,
                41.5, 27.6, 31.4, 32.9, 30.5,
                39.0, 44.9, 32.4, 34.7, 33.8,
                35.2, 29.7, 25.3, 24.4, 23.2,
                30.0, 34.4, 45.4, 41.9, 51.3,
                63.0, 52.9, 35.7, 37.9, 42.3,
                36.4, 36.8, 38.2, 36.0, 31.5,
                35.1, 29.6, 32.4, 42.3, 44.0,
                None, None, 42.0, 50.3, 25.6,
                41.2, 48.2, 40.2, 49.2, 40.2,
                41.5, 39.6, 39.5, 40.8, 43.5,
                43.7, 53.3, 36.8, None, 32.5,
                None, None, None, 33.7, 32.8
            ]
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.cache_dir}/gini_index.csv", index=False)
        return df


class CorrelationAnalyzer:
    """Analyzes correlations between economic freedom and quality of life indices."""

    def __init__(self, merged_data):
        self.data = merged_data
        self.results = {}

    def calculate_correlations(self):
        """Calculate Pearson correlations between economic freedom and all other indices."""
        print("\nCalculating correlations...")

        ef_col = 'Economic_Freedom_Score'
        indices = [
            ('GDP_Per_Capita', 'GDP per Capita (USD)', 'higher is better'),
            ('Life_Expectancy', 'Life Expectancy (years)', 'higher is better'),
            ('Infant_Mortality', 'Infant Mortality (per 1000)', 'lower is better'),
            ('Corruption_Score', 'Anti-Corruption Score', 'higher is better'),
            ('HDI', 'Human Development Index', 'higher is better'),
            ('Happiness_Score', 'Happiness Score', 'higher is better'),
            ('Peace_Score', 'Peace Score', 'higher is better'),
            ('Democracy_Score', 'Democracy Score', 'higher is better'),
            ('Social_Mobility_Score', 'Social Mobility Index', 'higher is better'),
            ('Purchasing_Power_Index', 'Purchasing Power', 'higher is better'),
            ('Military_Strength_Index', 'Military Strength', 'higher is better')
        ]

        for col, display_name, interpretation in indices:
            if col in self.data.columns:
                # Drop NaN values for this specific correlation
                valid_data = self.data[[ef_col, col]].dropna()

                if len(valid_data) > 10:  # Need enough data points
                    r, p_value = stats.pearsonr(valid_data[ef_col], valid_data[col])

                    # For infant mortality, flip the interpretation
                    if 'lower is better' in interpretation:
                        effective_correlation = -r  # Negative correlation with bad thing is good
                    else:
                        effective_correlation = r

                    self.results[col] = {
                        'display_name': display_name,
                        'correlation': r,
                        'effective_correlation': effective_correlation,
                        'p_value': p_value,
                        'n': len(valid_data),
                        'interpretation': interpretation,
                        'significant': p_value < 0.05
                    }

                    sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
                    print(f"  {display_name}: r = {r:.3f}{sig} (n={len(valid_data)})")

        return self.results

    def get_interpretation(self):
        """Generate textual interpretation of results."""
        interpretations = []

        for col, result in self.results.items():
            r = result['correlation']
            eff_r = result['effective_correlation']
            p = result['p_value']
            name = result['display_name']
            interp = result['interpretation']

            if p < 0.05:
                if eff_r > 0.5:
                    strength = "strong positive"
                    direction = "Countries with higher economic freedom tend to have significantly better"
                elif eff_r > 0.3:
                    strength = "moderate positive"
                    direction = "Countries with higher economic freedom tend to have somewhat better"
                elif eff_r > 0:
                    strength = "weak positive"
                    direction = "Countries with higher economic freedom tend to have slightly better"
                elif eff_r > -0.3:
                    strength = "weak negative"
                    direction = "Countries with higher economic freedom tend to have slightly worse"
                elif eff_r > -0.5:
                    strength = "moderate negative"
                    direction = "Countries with higher economic freedom tend to have somewhat worse"
                else:
                    strength = "strong negative"
                    direction = "Countries with higher economic freedom tend to have significantly worse"

                interpretations.append({
                    'index': name,
                    'strength': strength,
                    'direction': direction,
                    'r': r,
                    'p': p,
                    'favorable': eff_r > 0
                })

        return interpretations


class ReportGenerator:
    """Generates PDF report with visualizations."""

    def __init__(self, merged_data, correlation_results, output_file="economic_freedom_analysis.pdf"):
        self.data = merged_data
        self.results = correlation_results
        self.output_file = output_file

    def generate_report(self):
        """Generate the full PDF report."""
        print(f"\nGenerating PDF report: {self.output_file}")

        with PdfPages(self.output_file) as pdf:
            # Title page
            self._create_title_page(pdf)

            # Summary page
            self._create_summary_page(pdf)

            # Correlation matrix heatmap
            self._create_correlation_heatmap(pdf)

            # Individual scatter plots
            self._create_scatter_plots(pdf)

            # Regional analysis
            self._create_regional_analysis(pdf)

            # Conclusions page
            self._create_conclusions_page(pdf)

        print(f"Report saved to: {self.output_file}")

    def _create_title_page(self, pdf):
        """Create title page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        # Title
        ax.text(0.5, 0.7, 'Economic Freedom vs Quality of Life',
                fontsize=28, fontweight='bold', ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.6, 'A Statistical Analysis',
                fontsize=20, ha='center', transform=ax.transAxes)

        ax.text(0.5, 0.45, 'Data Sources:', fontsize=14, fontweight='bold',
                ha='center', transform=ax.transAxes)

        sources = [
            '• Heritage Foundation Economic Freedom Index (2024)',
            '• World Bank Development Indicators',
            '• Transparency International Corruption Perceptions Index',
            '• UN Human Development Index',
            '• World Happiness Report',
            '• Global Peace Index',
            '• Economist Intelligence Unit Democracy Index'
        ]

        for i, source in enumerate(sources):
            ax.text(0.5, 0.38 - i*0.04, source, fontsize=11, ha='center', transform=ax.transAxes)

        ax.text(0.5, 0.08, f'Analysis of {len(self.data)} countries',
                fontsize=12, style='italic', ha='center', transform=ax.transAxes)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_summary_page(self, pdf):
        """Create summary statistics page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        ax.text(0.5, 0.95, 'Summary of Correlations with Economic Freedom',
                fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)

        ax.text(0.5, 0.88, 'Higher Economic Freedom is Associated With:',
                fontsize=14, ha='center', transform=ax.transAxes)

        # Sort results by effective correlation
        sorted_results = sorted(self.results.items(),
                               key=lambda x: x[1]['effective_correlation'],
                               reverse=True)

        y_pos = 0.80
        for col, result in sorted_results:
            r = result['correlation']
            eff_r = result['effective_correlation']
            p = result['p_value']
            name = result['display_name']
            interp = result['interpretation']

            # Determine color and symbol based on effective correlation
            if eff_r > 0.3:
                color = '#2E7D32'  # Dark green
                symbol = '✓✓'
            elif eff_r > 0:
                color = '#81C784'  # Light green
                symbol = '✓'
            elif eff_r > -0.3:
                color = '#E57373'  # Light red
                symbol = '✗'
            else:
                color = '#C62828'  # Dark red
                symbol = '✗✗'

            sig_text = ""
            if p < 0.001:
                sig_text = "(p < 0.001) ***"
            elif p < 0.01:
                sig_text = "(p < 0.01) **"
            elif p < 0.05:
                sig_text = "(p < 0.05) *"
            else:
                sig_text = "(not significant)"
                color = '#757575'
                symbol = '—'

            # Display direction text
            if 'lower is better' in interp:
                direction_text = "LOWER" if r < 0 else "higher"
            else:
                direction_text = "HIGHER" if r > 0 else "lower"

            text = f"{symbol} {name}: r = {r:.3f} → {direction_text} {interp.replace('higher is better', '').replace('lower is better', '')} {sig_text}"

            ax.text(0.1, y_pos, text, fontsize=11, color=color,
                   transform=ax.transAxes, fontweight='bold' if p < 0.05 else 'normal')
            y_pos -= 0.06

        # Legend
        ax.text(0.1, 0.15, 'Interpretation Guide:', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.1, 0.10, '✓✓ Strong positive association (r > 0.3)', fontsize=10, color='#2E7D32', transform=ax.transAxes)
        ax.text(0.1, 0.06, '✓ Weak positive association (0 < r < 0.3)', fontsize=10, color='#81C784', transform=ax.transAxes)
        ax.text(0.5, 0.10, '✗ Weak negative association (-0.3 < r < 0)', fontsize=10, color='#E57373', transform=ax.transAxes)
        ax.text(0.5, 0.06, '✗✗ Strong negative association (r < -0.3)', fontsize=10, color='#C62828', transform=ax.transAxes)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_correlation_heatmap(self, pdf):
        """Create correlation heatmap."""
        fig, ax = plt.subplots(figsize=(11, 8.5))

        # Select numeric columns for correlation
        numeric_cols = ['Economic_Freedom_Score', 'GDP_Per_Capita', 'Life_Expectancy',
                       'Corruption_Score', 'HDI', 'Happiness_Score',
                       'Peace_Score', 'Democracy_Score']

        available_cols = [col for col in numeric_cols if col in self.data.columns]

        corr_matrix = self.data[available_cols].corr()

        # Rename for display
        rename_dict = {
            'Economic_Freedom_Score': 'Econ Freedom',
            'GDP_Per_Capita': 'GDP/Capita',
            'Life_Expectancy': 'Life Expect.',
            'Corruption_Score': 'Anti-Corrupt.',
            'HDI': 'HDI',
            'Happiness_Score': 'Happiness',
            'Peace_Score': 'Peace',
            'Democracy_Score': 'Democracy'
        }
        corr_matrix = corr_matrix.rename(index=rename_dict, columns=rename_dict)

        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   center=0, vmin=-1, vmax=1, mask=mask, ax=ax,
                   annot_kws={'size': 10})

        ax.set_title('Correlation Matrix: Economic Freedom & Quality of Life Indices',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_scatter_plots(self, pdf):
        """Create scatter plots for each index vs economic freedom."""
        indices = [
            ('GDP_Per_Capita', 'GDP per Capita (USD)', True),
            ('Life_Expectancy', 'Life Expectancy (years)', True),
            ('Corruption_Score', 'Anti-Corruption Score (0-100)', True),
            ('HDI', 'Human Development Index', True),
            ('Happiness_Score', 'Happiness Score', True),
            ('Peace_Score', 'Peace Score', True),
            ('Democracy_Score', 'Democracy Score', True),
            ('Infant_Mortality', 'Infant Mortality (per 1000 births)', False)
        ]

        for i in range(0, len(indices), 2):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            for j, ax in enumerate(axes):
                if i + j < len(indices):
                    col, name, higher_better = indices[i + j]

                    if col in self.data.columns:
                        valid_data = self.data[['Economic_Freedom_Score', col, 'Country']].dropna()

                        # Scatter plot
                        ax.scatter(valid_data['Economic_Freedom_Score'],
                                  valid_data[col], alpha=0.6, s=50)

                        # Add trend line
                        z = np.polyfit(valid_data['Economic_Freedom_Score'],
                                      valid_data[col], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(valid_data['Economic_Freedom_Score'].min(),
                                            valid_data['Economic_Freedom_Score'].max(), 100)
                        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
                               label='Trend line')

                        # Calculate correlation
                        r, p_val = stats.pearsonr(valid_data['Economic_Freedom_Score'],
                                                  valid_data[col])

                        # Determine if this is "good" or "bad" correlation
                        if higher_better:
                            is_good = r > 0
                        else:
                            is_good = r < 0

                        color = 'green' if is_good else 'red'
                        sig_stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))

                        ax.set_xlabel('Economic Freedom Score', fontsize=11)
                        ax.set_ylabel(name, fontsize=11)
                        ax.set_title(f'Economic Freedom vs {name}\nr = {r:.3f}{sig_stars}',
                                    fontsize=12, fontweight='bold')

                        # Add interpretation box
                        if p_val < 0.05:
                            if is_good:
                                interp_text = "↑ Economic Freedom\n= Better outcomes"
                            else:
                                interp_text = "↑ Economic Freedom\n= Worse outcomes"
                        else:
                            interp_text = "No significant\nrelationship"
                            color = 'gray'

                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                        ax.text(0.05, 0.95, interp_text, transform=ax.transAxes,
                               fontsize=10, verticalalignment='top', bbox=props, color=color)

                        # Label some notable countries
                        notable = ['United States', 'China', 'Singapore', 'Venezuela',
                                  'Norway', 'Cuba', 'Switzerland', 'North Korea']
                        for _, row in valid_data.iterrows():
                            if row['Country'] in notable:
                                ax.annotate(row['Country'][:10],
                                           (row['Economic_Freedom_Score'], row[col]),
                                           fontsize=7, alpha=0.7)
                else:
                    ax.axis('off')

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    def _create_regional_analysis(self, pdf):
        """Create regional analysis page."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Box plot by region
        if 'Region' in self.data.columns:
            region_order = self.data.groupby('Region')['Economic_Freedom_Score'].median().sort_values(ascending=False).index

            ax = axes[0]
            self.data.boxplot(column='Economic_Freedom_Score', by='Region', ax=ax)
            ax.set_title('Economic Freedom by Region', fontsize=12, fontweight='bold')
            ax.set_xlabel('Region')
            ax.set_ylabel('Economic Freedom Score')
            plt.suptitle('')  # Remove automatic title
            ax.tick_params(axis='x', rotation=45)

        # Average indices by economic freedom quartile
        ax = axes[1]
        self.data['EF_Quartile'] = pd.qcut(self.data['Economic_Freedom_Score'],
                                           q=4, labels=['Least Free', 'Less Free',
                                                       'More Free', 'Most Free'])

        indices_to_compare = ['HDI', 'Happiness_Score', 'Peace_Score', 'Life_Expectancy']
        available_indices = [idx for idx in indices_to_compare if idx in self.data.columns]

        # Normalize for comparison
        comparison_data = []
        for idx in available_indices:
            for quartile in ['Least Free', 'Less Free', 'More Free', 'Most Free']:
                val = self.data[self.data['EF_Quartile'] == quartile][idx].mean()
                max_val = self.data[idx].max()
                min_val = self.data[idx].min()
                normalized = (val - min_val) / (max_val - min_val) * 100
                comparison_data.append({
                    'Index': idx.replace('_', ' '),
                    'Quartile': quartile,
                    'Value': normalized
                })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_pivot = comparison_df.pivot(index='Index', columns='Quartile', values='Value')
        comparison_pivot = comparison_pivot[['Least Free', 'Less Free', 'More Free', 'Most Free']]

        comparison_pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Normalized Quality of Life by Economic Freedom Quartile',
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Score (0-100)')
        ax.set_xlabel('')
        ax.legend(title='Economic Freedom', loc='upper left')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_conclusions_page(self, pdf):
        """Create conclusions page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        ax.text(0.5, 0.95, 'Conclusions', fontsize=24, fontweight='bold',
               ha='center', transform=ax.transAxes)

        # Count positive and negative correlations
        sig_positive = sum(1 for r in self.results.values()
                          if r['effective_correlation'] > 0 and r['p_value'] < 0.05)
        sig_negative = sum(1 for r in self.results.values()
                          if r['effective_correlation'] < 0 and r['p_value'] < 0.05)
        total_sig = sig_positive + sig_negative

        conclusions = [
            f"OVERALL FINDING:",
            f"",
            f"Out of {len(self.results)} quality-of-life indices analyzed:",
            f"",
            f"   • {sig_positive} show POSITIVE correlation with economic freedom",
            f"     (higher economic freedom = better outcomes)",
            f"",
            f"   • {sig_negative} show NEGATIVE correlation with economic freedom",
            f"     (higher economic freedom = worse outcomes)",
            f"",
            f"   • {len(self.results) - total_sig} show no statistically significant relationship",
            f"",
            f"",
        ]

        if sig_positive > sig_negative:
            conclusions.extend([
                "INTERPRETATION:",
                "",
                "The data suggests that greater economic freedom is associated",
                "with better outcomes across most measured quality-of-life indices.",
                "",
                "Countries with higher economic freedom scores tend to have:",
            ])

            # List positive correlations
            for col, result in sorted(self.results.items(),
                                     key=lambda x: x[1]['effective_correlation'],
                                     reverse=True):
                if result['effective_correlation'] > 0 and result['p_value'] < 0.05:
                    conclusions.append(f"   • Higher {result['display_name']}")

        elif sig_negative > sig_positive:
            conclusions.extend([
                "INTERPRETATION:",
                "",
                "The data suggests that greater economic freedom is associated",
                "with worse outcomes across most measured quality-of-life indices.",
            ])
        else:
            conclusions.extend([
                "INTERPRETATION:",
                "",
                "The data shows mixed results with no clear overall trend.",
            ])

        conclusions.extend([
            "",
            "",
            "IMPORTANT CAVEATS:",
            "",
            "• Correlation does not imply causation",
            "• Many confounding variables exist (history, geography, culture)",
            "• Economic freedom indices have methodological limitations",
            "• Data availability varies by country",
        ])

        y_pos = 0.85
        for line in conclusions:
            if line.startswith("OVERALL") or line.startswith("INTERPRETATION") or line.startswith("IMPORTANT"):
                ax.text(0.1, y_pos, line, fontsize=12, fontweight='bold', transform=ax.transAxes)
            else:
                ax.text(0.1, y_pos, line, fontsize=11, transform=ax.transAxes)
            y_pos -= 0.028

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run the analysis."""
    print("=" * 60)
    print("Economic Freedom vs Quality of Life Analysis")
    print("=" * 60)

    # Fetch data
    fetcher = DataFetcher()

    economic_freedom = fetcher.fetch_heritage_economic_freedom()
    world_bank = fetcher.fetch_world_bank_data()
    corruption = fetcher.fetch_corruption_index()
    hdi = fetcher.fetch_human_development_index()
    happiness = fetcher.fetch_happiness_index()
    peace = fetcher.fetch_peace_index()
    democracy = fetcher.fetch_democracy_index()
    social_mobility = fetcher.fetch_social_mobility_index()

    # Merge all datasets
    print("\nMerging datasets...")

    merged = economic_freedom.copy()

    datasets = [
        (world_bank, 'Country'),
        (corruption, 'Country'),
        (hdi, 'Country'),
        (happiness, 'Country'),
        (peace, 'Country'),
        (democracy, 'Country'),
        (social_mobility, 'Country')
    ]

    for df, key in datasets:
        if df is not None and len(df) > 0:
            # Drop duplicate columns before merging
            cols_to_use = [key] + [c for c in df.columns if c not in merged.columns]
            merged = merged.merge(df[cols_to_use], on=key, how='left')

    print(f"  Merged dataset: {len(merged)} countries")

    # Calculate correlations
    analyzer = CorrelationAnalyzer(merged)
    results = analyzer.calculate_correlations()

    # Generate report
    report = ReportGenerator(merged, results)
    report.generate_report()

    # Print summary to console
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)

    sig_positive = sum(1 for r in results.values()
                      if r['effective_correlation'] > 0 and r['p_value'] < 0.05)
    sig_negative = sum(1 for r in results.values()
                      if r['effective_correlation'] < 0 and r['p_value'] < 0.05)

    print(f"\nStatistically significant correlations (p < 0.05):")
    print(f"  • Positive (freedom = better): {sig_positive}")
    print(f"  • Negative (freedom = worse):  {sig_negative}")

    if sig_positive > sig_negative:
        print(f"\n→ The data SUPPORTS the claim that economic freedom")
        print(f"  correlates with better quality-of-life outcomes.")
    elif sig_negative > sig_positive:
        print(f"\n→ The data CONTRADICTS the claim that economic freedom")
        print(f"  correlates with better quality-of-life outcomes.")
    else:
        print(f"\n→ The data shows MIXED results regarding economic freedom")
        print(f"  and quality-of-life outcomes.")

    print(f"\nFull report saved to: economic_freedom_analysis.pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()
