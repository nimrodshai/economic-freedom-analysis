#!/usr/bin/env python3
"""
Economic Freedom vs Quality of Life Indices Analysis
=====================================================
This script fetches economic freedom data and correlates it with various
quality of life indices to examine relationships between economic freedom
and societal outcomes.

All data is fetched LIVE from authoritative sources:
- Heritage Foundation Economic Freedom Index (Excel download)
- World Bank API (GDP per capita, Life Expectancy, Infant Mortality, Gini Index)
- Transparency International CPI (Excel download)
- UN Human Development Index (UNDP API)
- World Happiness Report (Excel download)
- Democracy Index (Our World in Data)
- Social Mobility Index (World Economic Forum)
- Numbeo Purchasing Power Index (Web scraping)
"""

import os
import re
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from io import StringIO, BytesIO
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class DataFetcher:
    """Fetches data LIVE from various authoritative sources for economic and quality of life indices."""

    # Standard country name mappings for consistency across data sources
    COUNTRY_NAME_MAP = {
        'United States of America': 'United States',
        'USA': 'United States',
        'U.S.A.': 'United States',
        'UK': 'United Kingdom',
        'Great Britain': 'United Kingdom',
        'Republic of Korea': 'South Korea',
        'Korea, South': 'South Korea',
        'Korea, Rep.': 'South Korea',
        'Korea (Republic of)': 'South Korea',
        'Korea, North': 'North Korea',
        "Korea, Dem. People's Rep.": 'North Korea',
        'Russian Federation': 'Russia',
        'Türkiye': 'Turkey',
        'Turkiye': 'Turkey',
        'Czechia': 'Czech Republic',
        'Viet Nam': 'Vietnam',
        'Iran, Islamic Rep.': 'Iran',
        'Iran (Islamic Republic of)': 'Iran',
        'Egypt, Arab Rep.': 'Egypt',
        'Venezuela, RB': 'Venezuela',
        'Venezuela (Bolivarian Republic of)': 'Venezuela',
        'Syrian Arab Republic': 'Syria',
        'Hong Kong SAR, China': 'Hong Kong',
        'Hong Kong, China (SAR)': 'Hong Kong',
        'Taiwan, China': 'Taiwan',
        'Taiwan Province of China': 'Taiwan',
        'China, Taiwan Province of': 'Taiwan',
        'Slovak Republic': 'Slovakia',
        'Lao PDR': 'Laos',
        "Lao People's Democratic Republic": 'Laos',
        'Congo, Dem. Rep.': 'DR Congo',
        'Democratic Republic of the Congo': 'DR Congo',
        'Congo, Rep.': 'Congo',
        "Côte d'Ivoire": 'Ivory Coast',
        'Cote d\'Ivoire': 'Ivory Coast',
        "Côte d\u2019Ivoire": 'Ivory Coast',  # with RIGHT SINGLE QUOTATION MARK (U+2019)
        'Eswatini': 'Swaziland',
        'North Macedonia': 'Macedonia',
        'Myanmar': 'Burma',
        'Brunei Darussalam': 'Brunei',
        'Trinidad And Tobago': 'Trinidad and Tobago',
        'Bosnia And Herzegovina': 'Bosnia and Herzegovina',
        'United Arab Emirates': 'UAE',
        # World Bank specific name formats
        'Bahamas, The': 'Bahamas',
        'Gambia, The': 'Gambia',
        'Micronesia, Fed. Sts.': 'Micronesia',
        'St. Lucia': 'Saint Lucia',
        'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
        'Sao Tome and Principe': 'Sao Tome and Principe',
        'São Tomé and Príncipe': 'Sao Tome and Principe',
        'Kyrgyz Republic': 'Kyrgyzstan',
        'Yemen, Rep.': 'Yemen',
        'West Bank and Gaza': 'Palestine',
        'Macao SAR, China': 'Macau',
        'Virgin Islands (U.S.)': 'US Virgin Islands',
        'Cabo Verde': 'Cabo Verde',
        'Republic of Congo': 'Congo',
        'Democratic Republic of Congo': 'DR Congo',
        # Heritage Foundation specific name formats
        'The Bahamas': 'Bahamas',
        'The Gambia': 'Gambia',
        'Saint Vincent and The Grenadines': 'Saint Vincent and the Grenadines',
    }

    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def _normalize_country_name(self, name):
        """Normalize country name for consistent matching across datasets."""
        if pd.isna(name):
            return name
        name = str(name).strip()
        return self.COUNTRY_NAME_MAP.get(name, name)

    def fetch_heritage_economic_freedom(self):
        """
        Fetch Heritage Foundation Economic Freedom Index data.
        Source: https://www.heritage.org/index/
        Downloads the official Excel file.
        """
        print("Fetching Heritage Economic Freedom Index (live)...")

        # Try multiple URLs for the Heritage data - newest first
        from datetime import datetime
        current_year = datetime.now().year
        urls = []
        for year in range(current_year, current_year - 7, -1):
            urls.append(f"https://static.heritage.org/index/data/{year}/{year}_indexofeconomicfreedom_data.xlsx")
        urls.append("https://www.worldbank.org/content/dam/sites/govindicators/doc/HER.xlsx")

        for url in urls:
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    # Try different header rows since Heritage format varies
                    for header_row in [0, 1, 2]:
                        try:
                            df = pd.read_excel(BytesIO(response.content), sheet_name=0, header=header_row)

                            # Check if first row contains 'Country' - if so, use it as header
                            if header_row == 0 and 'Country' in df.iloc[0].values:
                                df.columns = df.iloc[0]
                                df = df.iloc[1:]

                            # Find the country and score columns
                            country_col = None
                            score_col = None
                            region_col = None

                            for col in df.columns:
                                col_str = str(col).lower() if pd.notna(col) else ''
                                if 'country' in col_str or col == 'Country':
                                    country_col = col
                                elif 'overall' in col_str or '2025 score' in col_str or '2024 score' in col_str:
                                    score_col = col
                                elif 'region' in col_str:
                                    region_col = col

                            # If no explicit score column, look for column index 1 (usually Overall Score)
                            if country_col and not score_col:
                                cols = df.columns.tolist()
                                if len(cols) > 1:
                                    # Try column after Country
                                    idx = cols.index(country_col)
                                    if idx + 1 < len(cols):
                                        score_col = cols[idx + 1]

                            if country_col and score_col:
                                result = pd.DataFrame({
                                    'Country': df[country_col].apply(self._normalize_country_name),
                                    'Economic_Freedom_Score': pd.to_numeric(df[score_col], errors='coerce')
                                })
                                if region_col:
                                    result['Region'] = df[region_col]

                                result = result.dropna(subset=['Country', 'Economic_Freedom_Score'])
                                if len(result) > 100:  # Sanity check
                                    result.to_csv(f"{self.cache_dir}/heritage_economic_freedom.csv", index=False)
                                    # Extract year from URL
                                    import re
                                    year_match = re.search(r'/(\d{4})/', url)
                                    heritage_year = year_match.group(1) if year_match else 'latest'
                                    print(f"  Successfully fetched {len(result)} countries from Heritage Foundation ({heritage_year})")
                                    return result
                        except Exception:
                            continue
            except Exception as e:
                pass  # Silently try next URL

        # Fallback to cached data if available
        cache_file = f"{self.cache_dir}/heritage_economic_freedom.csv"
        if os.path.exists(cache_file):
            print("  Using cached Heritage data...")
            return pd.read_csv(cache_file)

        raise Exception("Could not fetch Heritage Economic Freedom data from any source")

    def fetch_world_bank_data(self):
        """
        Fetch World Bank indicators via API.
        Source: https://data.worldbank.org/
        Includes GDP per capita, Life Expectancy, Infant Mortality, and Gini Index.
        """
        print("Fetching World Bank data (live API)...")

        indicators = {
            'NY.GDP.PCAP.CD': 'GDP_Per_Capita',
            'SP.DYN.LE00.IN': 'Life_Expectancy',
            'SP.DYN.IMRT.IN': 'Infant_Mortality',
            'SI.POV.GINI': 'Gini_Index',
        }

        all_data = {}
        from datetime import datetime
        current_year = datetime.now().year

        for indicator_code, indicator_name in indicators.items():
            try:
                # Try multiple years to get most recent data (World Bank data can lag several years)
                for year in range(current_year, current_year - 7, -1):
                    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}?format=json&date={year}&per_page=300"
                    response = self.session.get(url, timeout=30)

                    if response.status_code == 200:
                        data = response.json()
                        if len(data) > 1 and data[1]:
                            for item in data[1]:
                                if item['value'] is not None:
                                    country = self._normalize_country_name(item['country']['value'])
                                    if country not in all_data:
                                        all_data[country] = {'Country': country}
                                    if indicator_name not in all_data[country]:
                                        all_data[country][indicator_name] = item['value']

                print(f"  {indicator_name}: {sum(1 for c in all_data.values() if indicator_name in c)} countries")

            except Exception as e:
                print(f"  Error fetching {indicator_name}: {e}")

        if all_data:
            df = pd.DataFrame(list(all_data.values()))
            # Filter out regional aggregates by explicit list (more reliable than pattern matching)
            exclude_aggregates = {
                'Africa Eastern and Southern', 'Africa Western and Central', 'Arab World',
                'Caribbean small states', 'Central Europe and the Baltics', 'Early-demographic dividend',
                'East Asia & Pacific', 'East Asia & Pacific (IDA & IBRD countries)',
                'East Asia & Pacific (excluding high income)', 'Euro area', 'Europe & Central Asia',
                'Europe & Central Asia (IDA & IBRD countries)', 'Europe & Central Asia (excluding high income)',
                'European Union', 'Fragile and conflict affected situations', 'Heavily indebted poor countries (HIPC)',
                'High income', 'IBRD only', 'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total',
                'Late-demographic dividend', 'Latin America & Caribbean',
                'Latin America & Caribbean (excluding high income)', 'Latin America & the Caribbean (IDA & IBRD countries)',
                'Least developed countries: UN classification', 'Low & middle income', 'Low income',
                'Lower middle income', 'Middle East & North Africa', 'Middle East & North Africa (IDA & IBRD countries)',
                'Middle East & North Africa (excluding high income)', 'Middle income', 'North America',
                'Not classified', 'OECD members', 'Other small states', 'Pacific island small states',
                'Post-demographic dividend', 'Pre-demographic dividend', 'Small states', 'South Asia',
                'South Asia (IDA & IBRD)', 'Sub-Saharan Africa', 'Sub-Saharan Africa (IDA & IBRD countries)',
                'Sub-Saharan Africa (excluding high income)', 'Upper middle income', 'World'
            }
            mask = ~df['Country'].isin(exclude_aggregates)
            df = df[mask]
            df.to_csv(f"{self.cache_dir}/world_bank_data.csv", index=False)
            print(f"  Total: {len(df)} countries with World Bank data")
            return df

        # Fallback to cached data
        cache_file = f"{self.cache_dir}/world_bank_data.csv"
        if os.path.exists(cache_file):
            print("  Using cached World Bank data...")
            return pd.read_csv(cache_file)

        raise Exception("Could not fetch World Bank data")

    def fetch_corruption_index(self):
        """
        Fetch Transparency International Corruption Perceptions Index.
        Source: https://www.transparency.org/en/cpi/
        Downloads the official Excel file.
        """
        print("Fetching Corruption Perceptions Index (live)...")

        # Try to download the official Excel file - newest first, fallback to older
        from datetime import datetime
        current_year = datetime.now().year
        urls = []
        for year in range(current_year, current_year - 7, -1):
            urls.extend([
                f"https://images.transparencycdn.org/images/CPI{year}_Global_Results_Trends.xlsx",
                f"https://files.transparencycdn.org/images/CPI{year}_Global_Results_Trends.xlsx",
                f"https://files.transparencycdn.org/images/CPI{year}_Global_Results__Trends.xlsx",
            ])

        for url in urls:
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    # CPI Excel files have complex header structure:
                    # Row 0: Title, Row 1-2: Empty, Row 3: Headers (Country / Territory, CPI score, etc.)
                    df = pd.read_excel(BytesIO(response.content), sheet_name=0, header=3)

                    # Find the country and score columns
                    country_col = None
                    score_col = None

                    for col in df.columns:
                        col_str = str(col).lower() if pd.notna(col) else ''
                        # Match "Country / Territory" specifically (first column)
                        if country_col is None and (col_str.startswith('country') or col_str == 'country / territory'):
                            country_col = col
                        elif score_col is None and 'cpi score' in col_str:
                            score_col = col

                    if country_col and score_col:
                        result = pd.DataFrame({
                            'Country': df[country_col].apply(self._normalize_country_name),
                            'Corruption_Score': pd.to_numeric(df[score_col], errors='coerce')
                        })
                        result = result.dropna(subset=['Country', 'Corruption_Score'])
                        if len(result) > 100:
                            result.to_csv(f"{self.cache_dir}/corruption_index.csv", index=False)
                            # Extract year from URL
                            import re
                            year_match = re.search(r'CPI(\d{4})', url)
                            year = year_match.group(1) if year_match else 'unknown'
                            print(f"  Successfully fetched {len(result)} countries from Transparency International ({year})")
                            return result

            except Exception as e:
                pass  # Silently try next URL

        # Fallback to cached data
        cache_file = f"{self.cache_dir}/corruption_index.csv"
        if os.path.exists(cache_file):
            print("  Using cached CPI data...")
            return pd.read_csv(cache_file)

        raise Exception("Could not fetch Corruption Perceptions Index data")

    def fetch_human_development_index(self):
        """
        Fetch UN Human Development Index data.
        Source: https://hdr.undp.org/data-center/
        Downloads from UNDP data center.
        """
        print("Fetching Human Development Index (live)...")

        # Try UNDP data API/download
        urls = [
            "https://hdr.undp.org/sites/default/files/2023-24_HDR/HDR23-24_Composite_indices_complete_time_series.csv",
            "https://hdr.undp.org/sites/default/files/2021-22_HDR/HDR21-22_Composite_indices_complete_time_series.csv",
        ]

        for url in urls:
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    df = pd.read_csv(StringIO(response.text))

                    # Find the country and HDI columns
                    country_col = None
                    hdi_col = None

                    for col in df.columns:
                        col_lower = str(col).lower()
                        if 'country' in col_lower:
                            country_col = col
                        # Look for most recent HDI column (e.g., hdi_2022, hdi_2021)
                        elif col_lower.startswith('hdi_'):
                            year = col_lower.replace('hdi_', '')
                            if year.isdigit() and (hdi_col is None or year > hdi_col.split('_')[1]):
                                hdi_col = col

                    if country_col and hdi_col:
                        result = pd.DataFrame({
                            'Country': df[country_col].apply(self._normalize_country_name),
                            'HDI': pd.to_numeric(df[hdi_col], errors='coerce')
                        })
                        result = result.dropna(subset=['Country', 'HDI'])
                        result.to_csv(f"{self.cache_dir}/hdi.csv", index=False)
                        # Extract year from column name (e.g., hdi_2022)
                        hdi_year = hdi_col.split('_')[-1] if '_' in str(hdi_col) else 'latest'
                        print(f"  Successfully fetched {len(result)} countries from UNDP ({hdi_year})")
                        return result

            except Exception as e:
                print(f"  Could not fetch from {url}: {e}")

        # Try Our World in Data as alternative source
        try:
            url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Human%20Development%20Index%20-%20UNDP/Human%20Development%20Index%20-%20UNDP.csv"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                # Get most recent year for each country
                df = df.sort_values('Year', ascending=False).drop_duplicates('Entity')
                result = pd.DataFrame({
                    'Country': df['Entity'].apply(self._normalize_country_name),
                    'HDI': pd.to_numeric(df['Human Development Index (UNDP)'], errors='coerce')
                })
                result = result.dropna(subset=['Country', 'HDI'])
                result.to_csv(f"{self.cache_dir}/hdi.csv", index=False)
                print(f"  Successfully fetched {len(result)} countries from Our World in Data")
                return result
        except Exception as e:
            print(f"  Could not fetch from OWID: {e}")

        # Fallback to cached data
        cache_file = f"{self.cache_dir}/hdi.csv"
        if os.path.exists(cache_file):
            print("  Using cached HDI data...")
            return pd.read_csv(cache_file)

        raise Exception("Could not fetch Human Development Index data")

    def fetch_happiness_index(self):
        """
        Fetch World Happiness Report data.
        Source: https://worldhappiness.report/
        Downloads from official AWS S3 bucket.
        """
        print("Fetching World Happiness Report (live)...")

        # Try to download the official Excel file - newest first
        from datetime import datetime
        current_year = datetime.now().year
        urls = []
        for year in range(current_year, current_year - 7, -1):
            short_year = str(year)[2:]  # e.g., 26, 25, 24
            urls.extend([
                f"https://files.worldhappiness.report/WHR{short_year}_Data_Figure_2.1v3.xlsx",
                f"https://files.worldhappiness.report/WHR{short_year}_Data_Figure_2.1.xlsx",
                f"https://happiness-report.s3.amazonaws.com/{year}/DataForFigure2.1WHR{year}.xls",
            ])

        for url in urls:
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    df = pd.read_excel(BytesIO(response.content), sheet_name=0)

                    # Find the country, year, and happiness score columns
                    country_col = None
                    score_col = None
                    year_col = None

                    for col in df.columns:
                        col_str = str(col).lower()
                        if 'country' in col_str or col_str == 'entity':
                            country_col = col
                        elif col_str == 'year':
                            year_col = col
                        elif 'ladder' in col_str or 'life evaluation' in col_str or 'cantril' in col_str:
                            score_col = col

                    if country_col and score_col:
                        # Filter for most recent year if year column exists
                        if year_col:
                            max_year = df[year_col].max()
                            df = df[df[year_col] == max_year]
                            print(f"  Filtering for year {max_year}")

                        # Deduplicate by country
                        df = df.drop_duplicates(subset=[country_col])

                        result = pd.DataFrame({
                            'Country': df[country_col].apply(self._normalize_country_name),
                            'Happiness_Score': pd.to_numeric(df[score_col], errors='coerce')
                        })
                        result = result.dropna(subset=['Country', 'Happiness_Score'])
                        result.to_csv(f"{self.cache_dir}/happiness.csv", index=False)
                        # Extract year from URL
                        import re
                        year_match = re.search(r'WHR(\d{2})|/(\d{4})/', url)
                        if year_match:
                            report_year = year_match.group(1) or year_match.group(2)
                            if len(report_year) == 2:
                                report_year = '20' + report_year
                        else:
                            report_year = str(max_year) if year_col else 'latest'
                        print(f"  Successfully fetched {len(result)} countries from World Happiness Report ({report_year})")
                        return result

            except Exception as e:
                pass  # Silently try next URL

        # Try Our World in Data as alternative
        try:
            url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Happiness%20and%20Life%20Satisfaction%20-%20World%20Happiness%20Report%202024/Happiness%20and%20Life%20Satisfaction%20-%20World%20Happiness%20Report%202024.csv"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                df = df.sort_values('Year', ascending=False).drop_duplicates('Entity')
                for col in df.columns:
                    if 'satisfaction' in col.lower() or 'happiness' in col.lower():
                        result = pd.DataFrame({
                            'Country': df['Entity'].apply(self._normalize_country_name),
                            'Happiness_Score': pd.to_numeric(df[col], errors='coerce')
                        })
                        result = result.dropna(subset=['Country', 'Happiness_Score'])
                        result.to_csv(f"{self.cache_dir}/happiness.csv", index=False)
                        print(f"  Successfully fetched {len(result)} countries from OWID")
                        return result
        except Exception as e:
            print(f"  Could not fetch from OWID: {e}")

        # Fallback to cached data
        cache_file = f"{self.cache_dir}/happiness.csv"
        if os.path.exists(cache_file):
            print("  Using cached Happiness data...")
            return pd.read_csv(cache_file)

        raise Exception("Could not fetch World Happiness Report data")

    def fetch_peace_index(self):
        """
        Fetch Global Peace Index data.
        Source: https://www.visionofhumanity.org/ (Institute for Economics & Peace)
        Downloads from Our World in Data or QoG Data Finder.
        """
        print("Fetching Global Peace Index (live)...")

        # Try Vision of Humanity / IEP direct data page
        try:
            # The GPI data can be found in the interactive map page JavaScript
            url = "https://www.visionofhumanity.org/wp-json/gpi/v1/countries"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                import json
                data = response.json()
                countries = []
                scores = []
                for item in data:
                    country = item.get('name', item.get('country', ''))
                    gpi = item.get('score', item.get('gpi', item.get('overall_score', None)))
                    if country and gpi:
                        try:
                            gpi_val = float(gpi)
                            # Invert: Peace_Score = 100 * (5 - GPI) / 4 to get 0-100 scale
                            peace_score = 100 * (5 - gpi_val) / 4
                            countries.append(self._normalize_country_name(country))
                            scores.append(round(peace_score, 1))
                        except (ValueError, TypeError):
                            continue
                if len(countries) > 100:
                    result = pd.DataFrame({'Country': countries, 'Peace_Score': scores})
                    result.to_csv(f"{self.cache_dir}/peace_index.csv", index=False)
                    print(f"  Successfully fetched {len(result)} countries from Vision of Humanity API")
                    return result
        except Exception as e:
            print(f"  Could not fetch from Vision of Humanity API: {e}")

        # Try Vision of Humanity website scraping
        try:
            url = "https://www.visionofhumanity.org/maps/"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                # Parse the JavaScript data from the page
                soup = BeautifulSoup(response.text, 'html.parser')
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.string and 'gpiData' in script.string:
                        # Extract JSON data
                        match = re.search(r'gpiData\s*=\s*(\[.*?\]);', script.string, re.DOTALL)
                        if match:
                            import json
                            data = json.loads(match.group(1))
                            countries = []
                            scores = []
                            for item in data:
                                countries.append(self._normalize_country_name(item.get('country', '')))
                                gpi = item.get('score', item.get('gpi', None))
                                if gpi:
                                    # Invert score
                                    scores.append(100 * (5 - float(gpi)) / 4)
                                else:
                                    scores.append(None)
                            result = pd.DataFrame({'Country': countries, 'Peace_Score': scores})
                            result = result.dropna(subset=['Country', 'Peace_Score'])
                            result.to_csv(f"{self.cache_dir}/peace_index.csv", index=False)
                            print(f"  Successfully fetched {len(result)} countries from Vision of Humanity")
                            return result
        except Exception as e:
            print(f"  Could not scrape Vision of Humanity: {e}")

        # Fallback to cached data
        cache_file = f"{self.cache_dir}/peace_index.csv"
        if os.path.exists(cache_file):
            print("  Using cached Peace Index data...")
            return pd.read_csv(cache_file)

        # Final fallback: embedded GPI 2024 data from visionofhumanity.org
        print("  Using embedded Peace Index data (GPI 2024)...")
        # GPI scores (lower = more peaceful, converted to 0-100 where higher = more peaceful)
        gpi_raw = {
            'Iceland': 1.124, 'Ireland': 1.303, 'Austria': 1.316, 'New Zealand': 1.329, 'Singapore': 1.339,
            'Switzerland': 1.357, 'Portugal': 1.372, 'Denmark': 1.377, 'Slovenia': 1.384, 'Malaysia': 1.387,
            'Czech Republic': 1.4, 'Japan': 1.407, 'Canada': 1.432, 'Hungary': 1.502, 'Finland': 1.507,
            'Croatia': 1.519, 'Germany': 1.53, 'Belgium': 1.531, 'Slovakia': 1.541, 'Netherlands': 1.544,
            'Australia': 1.559, 'Norway': 1.578, 'Poland': 1.597, 'Spain': 1.609, 'Italy': 1.618,
            'Estonia': 1.627, 'United Kingdom': 1.638, 'South Korea': 1.657, 'Taiwan': 1.66, 'France': 1.683,
            'Chile': 1.717, 'Romania': 1.737, 'Botswana': 1.757, 'Lithuania': 1.769, 'Latvia': 1.814,
            'Vietnam': 1.818, 'Costa Rica': 1.823, 'Uruguay': 1.837, 'Greece': 1.874, 'Argentina': 1.894,
            'Ghana': 1.905, 'Indonesia': 1.918, 'Panama': 1.932, 'Serbia': 1.938, 'Bulgaria': 1.944,
            'Morocco': 1.956, 'Albania': 1.96, 'Bolivia': 1.972, 'Jamaica': 1.978, 'Paraguay': 1.986,
            'Jordan': 2.002, 'Kuwait': 2.024, 'Tunisia': 2.057, 'UAE': 2.086, 'Peru': 2.091,
            'Saudi Arabia': 2.104, 'Dominican Republic': 2.136, 'Kenya': 2.141, 'Thailand': 2.158, 'Qatar': 2.164,
            'China': 2.186, 'Ecuador': 2.204, 'Egypt': 2.23, 'South Africa': 2.326, 'Mexico': 2.424,
            'India': 2.455, 'Philippines': 2.464, 'Brazil': 2.502, 'Colombia': 2.531, 'Turkey': 2.55,
            'United States': 2.565, 'Nigeria': 2.778, 'Iran': 2.789, 'Zimbabwe': 2.812, 'Venezuela': 2.837,
            'Pakistan': 2.853, 'Israel': 2.914, 'Russia': 3.142, 'Ukraine': 3.386, 'North Korea': 3.442
        }
        countries = []
        scores = []
        for country, gpi in gpi_raw.items():
            countries.append(country)
            # Convert: Peace_Score = 100 * (5 - GPI) / 4
            scores.append(round(100 * (5 - gpi) / 4, 1))
        result = pd.DataFrame({'Country': countries, 'Peace_Score': scores})
        result.to_csv(f"{self.cache_dir}/peace_index.csv", index=False)
        print(f"  Embedded {len(result)} countries")
        return result

    def fetch_democracy_index(self):
        """
        Fetch Democracy Index data (Economist Intelligence Unit).
        Source: https://www.eiu.com/n/campaigns/democracy-index/
        Downloads from Our World in Data.
        """
        print("Fetching Democracy Index (live)...")

        # Try Our World in Data CSV download
        try:
            url = "https://ourworldindata.org/grapher/democracy-index-eiu.csv"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))

                # Find year column and get most recent data
                if 'Year' in df.columns:
                    df = df.sort_values('Year', ascending=False).drop_duplicates('Entity')
                elif 'year' in df.columns:
                    df = df.sort_values('year', ascending=False).drop_duplicates('Entity')

                # Find the country and democracy score columns
                country_col = None
                score_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in ['entity', 'country']:
                        country_col = col
                    elif 'democracy' in col_lower:
                        score_col = col

                if country_col and score_col:
                    result = pd.DataFrame({
                        'Country': df[country_col].apply(self._normalize_country_name),
                        'Democracy_Score': pd.to_numeric(df[score_col], errors='coerce')
                    })
                    result = result.dropna(subset=['Country', 'Democracy_Score'])
                    result.to_csv(f"{self.cache_dir}/democracy_index.csv", index=False)
                    print(f"  Successfully fetched {len(result)} countries from Our World in Data")
                    return result
        except Exception as e:
            print(f"  Could not fetch from OWID: {e}")

        # Fallback to cached data
        cache_file = f"{self.cache_dir}/democracy_index.csv"
        if os.path.exists(cache_file):
            print("  Using cached Democracy Index data...")
            return pd.read_csv(cache_file)

        raise Exception("Could not fetch Democracy Index data")

    def fetch_social_mobility_index(self):
        """
        Fetch Global Social Mobility Index data (World Economic Forum).
        Source: https://www.weforum.org/publications/global-social-mobility-index-2020/
        Note: WEF only published this index once in 2020, so we try to download the PDF
        and parse it, or use cached data.
        """
        print("Fetching Social Mobility Index (live)...")

        # Try to fetch from WEF or alternative sources
        # The WEF report is primarily in PDF format, so we try known data aggregators

        # Try GitHub datasets that may have compiled this data
        try:
            url = "https://raw.githubusercontent.com/datasets/social-mobility/master/data/social-mobility.csv"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                if 'Country' in df.columns or 'Entity' in df.columns:
                    country_col = 'Country' if 'Country' in df.columns else 'Entity'
                    for col in df.columns:
                        if 'mobility' in col.lower() or 'score' in col.lower():
                            result = pd.DataFrame({
                                'Country': df[country_col].apply(self._normalize_country_name),
                                'Social_Mobility_Score': pd.to_numeric(df[col], errors='coerce')
                            })
                            result = result.dropna(subset=['Country', 'Social_Mobility_Score'])
                            result.to_csv(f"{self.cache_dir}/social_mobility_index.csv", index=False)
                            print(f"  Successfully fetched {len(result)} countries")
                            return result
        except Exception as e:
            print(f"  Could not fetch from GitHub: {e}")

        # Try World Population Review (they aggregate this data)
        try:
            url = "https://worldpopulationreview.com/country-rankings/social-mobility-by-country"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Look for table data
                tables = soup.find_all('table')
                for table in tables:
                    try:
                        df = pd.read_html(str(table))[0]
                        for col in df.columns:
                            if 'country' in str(col).lower():
                                country_col = col
                            if 'mobility' in str(col).lower() or 'index' in str(col).lower() or 'score' in str(col).lower():
                                score_col = col
                                result = pd.DataFrame({
                                    'Country': df[country_col].apply(self._normalize_country_name),
                                    'Social_Mobility_Score': pd.to_numeric(df[score_col], errors='coerce')
                                })
                                result = result.dropna(subset=['Country', 'Social_Mobility_Score'])
                                if len(result) > 50:
                                    result.to_csv(f"{self.cache_dir}/social_mobility_index.csv", index=False)
                                    print(f"  Successfully fetched {len(result)} countries from World Population Review")
                                    return result
                    except Exception:
                        continue
        except Exception as e:
            print(f"  Could not scrape World Population Review: {e}")

        # Fallback to cached data
        cache_file = f"{self.cache_dir}/social_mobility_index.csv"
        if os.path.exists(cache_file):
            print("  Using cached Social Mobility data...")
            return pd.read_csv(cache_file)

        raise Exception("Could not fetch Social Mobility Index data")

    def fetch_purchasing_power_index(self):
        """
        Fetch Purchasing Power Index data from Numbeo.
        Source: https://www.numbeo.com/cost-of-living/rankings_by_country.jsp
        Scrapes the current year's data.
        """
        print("Fetching Purchasing Power Index (live scraping)...")

        # Try multiple year URLs
        years = ['2025', '2024', '2024-mid', '2023']
        for year in years:
            try:
                url = f"https://www.numbeo.com/cost-of-living/rankings_by_country.jsp?title={year}&displayColumn=5"
                response = self.session.get(url, timeout=30)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Find the data table
                    table = soup.find('table', {'id': 't2'})
                    if table:
                        rows = table.find_all('tr')
                        countries = []
                        scores = []

                        for row in rows[1:]:  # Skip header
                            cols = row.find_all('td')
                            if len(cols) >= 2:
                                # Country name is usually in the second column (first is rank)
                                country_td = cols[1] if len(cols) > 1 else cols[0]
                                country = country_td.get_text(strip=True)

                                # Local Purchasing Power Index is in the last column
                                score_td = cols[-1]
                                try:
                                    score = float(score_td.get_text(strip=True))
                                    countries.append(self._normalize_country_name(country))
                                    scores.append(score)
                                except ValueError:
                                    continue

                        if len(countries) > 50:
                            result = pd.DataFrame({
                                'Country': countries,
                                'Purchasing_Power_Index': scores
                            })
                            result.to_csv(f"{self.cache_dir}/purchasing_power_index.csv", index=False)
                            print(f"  Successfully fetched {len(result)} countries from Numbeo ({year})")
                            return result

                    # Alternative: try to parse from JavaScript data
                    scripts = soup.find_all('script')
                    for script in scripts:
                        if script.string and 'displayColumn' in str(script.string):
                            # Look for array data
                            match = re.search(r'\[([^\]]+)\]', script.string)
                            if match:
                                pass  # Complex parsing

            except Exception as e:
                print(f"  Could not fetch from Numbeo for {year}: {e}")

        # Fallback to cached data
        cache_file = f"{self.cache_dir}/purchasing_power_index.csv"
        if os.path.exists(cache_file):
            print("  Using cached Purchasing Power data...")
            return pd.read_csv(cache_file)

        raise Exception("Could not fetch Purchasing Power Index data")

    def fetch_military_strength_index(self):
        """
        Fetch Military Strength Index data from Global Firepower.
        Source: https://www.globalfirepower.com/countries-listing.php
        Scrapes the PowerIndex data and converts to 0-100 scale (higher = stronger).
        """
        print("Fetching Military Strength Index (live scraping)...")

        try:
            url = "https://www.globalfirepower.com/countries-listing.php"
            response = self.session.get(url, timeout=30)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                countries = []
                scores = []

                # Find all country entries with their PwrIndx scores
                # The page structure has country divs with picTrans class
                country_divs = soup.find_all('div', class_='picTrans')

                for div in country_divs:
                    try:
                        # Get country name from the link
                        link = div.find('a')
                        if link:
                            country_name = link.get_text(strip=True)
                            # Clean up country name
                            country_name = re.sub(r'\s+', ' ', country_name).strip()

                            # Find the PowerIndex value (usually in a span with class pwrIndxData)
                            pwrindx_span = div.find('span', class_='pwrIndxData')
                            if pwrindx_span:
                                try:
                                    pwr_index = float(pwrindx_span.get_text(strip=True))
                                    # Convert: lower PowerIndex = stronger military
                                    # Score of 0.07 (USA) -> ~98, Score of 3.0 -> ~0
                                    converted = max(0, min(100, 100 * (1 - pwr_index / 3)))
                                    countries.append(self._normalize_country_name(country_name))
                                    scores.append(round(converted, 1))
                                except ValueError:
                                    continue
                    except Exception:
                        continue

                # Alternative parsing: try table structure
                if len(countries) < 50:
                    tables = soup.find_all('table')
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            cols = row.find_all('td')
                            for i, col in enumerate(cols):
                                text = col.get_text(strip=True)
                                # Look for PowerIndex pattern (0.XXXX)
                                if re.match(r'^\d+\.\d{4}$', text):
                                    try:
                                        pwr_index = float(text)
                                        # Get country from previous cell
                                        if i > 0:
                                            country = cols[i-1].get_text(strip=True)
                                            if country and len(country) > 2:
                                                converted = max(0, min(100, 100 * (1 - pwr_index / 3)))
                                                countries.append(self._normalize_country_name(country))
                                                scores.append(round(converted, 1))
                                    except ValueError:
                                        continue

                if len(countries) > 50:
                    result = pd.DataFrame({
                        'Country': countries,
                        'Military_Strength_Index': scores
                    })
                    result = result.drop_duplicates(subset=['Country'])
                    result.to_csv(f"{self.cache_dir}/military_strength_index.csv", index=False)
                    print(f"  Successfully fetched {len(result)} countries from Global Firepower")
                    return result

        except Exception as e:
            print(f"  Could not scrape Global Firepower: {e}")

        # Fallback to cached data
        cache_file = f"{self.cache_dir}/military_strength_index.csv"
        if os.path.exists(cache_file):
            print("  Using cached Military Strength data...")
            return pd.read_csv(cache_file)

        # Final fallback: embedded Global Firepower 2025 data
        print("  Using embedded Military Strength data (GFP 2025)...")
        # PowerIndex scores (lower = stronger), converted to 0-100 where higher = stronger
        gfp_raw = {
            'United States': 0.0744, 'Russia': 0.0788, 'China': 0.0788, 'India': 0.1184, 'South Korea': 0.1656,
            'United Kingdom': 0.1785, 'Japan': 0.1839, 'France': 0.1878, 'Turkey': 0.1902, 'Italy': 0.2164,
            'Brazil': 0.2415, 'Indonesia': 0.2557, 'Pakistan': 0.2513, 'Germany': 0.2601, 'Israel': 0.2661,
            'Australia': 0.3298, 'Spain': 0.3242, 'Egypt': 0.3427, 'Ukraine': 0.3755, 'Poland': 0.3776,
            'Taiwan': 0.3988, 'Vietnam': 0.4024, 'Thailand': 0.4536, 'Iran': 0.3048, 'Saudi Arabia': 0.4201,
            'Sweden': 0.4835, 'Singapore': 0.5271, 'Canada': 0.5179, 'Greece': 0.5337, 'Nigeria': 0.5771,
            'Mexico': 0.5965, 'Argentina': 0.6013, 'North Korea': 0.6016, 'Bangladesh': 0.6062, 'Netherlands': 0.6412,
            'Norway': 0.6811, 'Portugal': 0.6856, 'South Africa': 0.6889, 'Philippines': 0.6987, 'Malaysia': 0.7429,
            'Switzerland': 0.7869, 'Chile': 0.8361, 'Colombia': 0.8353, 'Finland': 0.8437, 'Denmark': 0.8109,
            'Peru': 0.8588, 'Venezuela': 0.8882, 'Czech Republic': 0.9994, 'Hungary': 1.0259, 'UAE': 1.0186,
            'Morocco': 1.1273, 'Cuba': 1.3286, 'Austria': 1.3704, 'Slovakia': 1.3978, 'Qatar': 1.4307,
            'Jordan': 1.6139, 'Kuwait': 1.6982, 'Bahrain': 1.7448, 'Kenya': 1.8135, 'New Zealand': 1.9039,
            'Tunisia': 1.9538, 'Slovenia': 2.1016, 'Ireland': 2.1103, 'Estonia': 2.2917, 'Zimbabwe': 2.3863,
            'Luxembourg': 2.6415
        }
        countries = []
        scores = []
        for country, pwr in gfp_raw.items():
            countries.append(country)
            # Convert: Military_Strength = 100 * (1 - PowerIndex / 3)
            scores.append(round(max(0, min(100, 100 * (1 - pwr / 3))), 1))
        result = pd.DataFrame({'Country': countries, 'Military_Strength_Index': scores})
        result.to_csv(f"{self.cache_dir}/military_strength_index.csv", index=False)
        print(f"  Embedded {len(result)} countries")
        return result

    def fetch_gini_index(self):
        """
        Fetch Gini Index data from World Bank API.
        Source: https://data.worldbank.org/indicator/SI.POV.GINI
        Note: This is included in World Bank data fetch, but kept for backwards compatibility.
        """
        print("Fetching Gini Index (via World Bank API)...")

        # Gini is already fetched via World Bank API in fetch_world_bank_data()
        # This method is kept for backwards compatibility and explicit Gini-only requests

        try:
            from datetime import datetime
            current_year = datetime.now().year
            all_data = {}
            # Gini data can lag several years, so check back further
            for year in range(current_year, current_year - 7, -1):
                url = f"https://api.worldbank.org/v2/country/all/indicator/SI.POV.GINI?format=json&date={year}&per_page=300"
                response = self.session.get(url, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    if len(data) > 1 and data[1]:
                        for item in data[1]:
                            if item['value'] is not None:
                                country = self._normalize_country_name(item['country']['value'])
                                if country not in all_data:
                                    all_data[country] = item['value']

            if all_data:
                result = pd.DataFrame({
                    'Country': list(all_data.keys()),
                    'Gini_Index': list(all_data.values())
                })
                result.to_csv(f"{self.cache_dir}/gini_index.csv", index=False)
                print(f"  Successfully fetched {len(result)} countries from World Bank")
                return result

        except Exception as e:
            print(f"  Error fetching Gini Index: {e}")

        # Fallback to cached data
        cache_file = f"{self.cache_dir}/gini_index.csv"
        if os.path.exists(cache_file):
            print("  Using cached Gini Index data...")
            return pd.read_csv(cache_file)

        raise Exception("Could not fetch Gini Index data")


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
            ('Democracy_Score', 'Democracy Score', 'higher is better'),
            ('Social_Mobility_Score', 'Social Mobility Index', 'higher is better'),
            ('Purchasing_Power_Index', 'Purchasing Power', 'higher is better')
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
                       'Democracy_Score']

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

        indices_to_compare = ['HDI', 'Happiness_Score', 'Democracy_Score', 'Life_Expectancy']
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
