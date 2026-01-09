# Economic Freedom vs Quality of Life Analysis

A data-driven analysis examining correlations between economic freedom and various quality-of-life indicators across countries worldwide.

## Live Demo

Visit the live analysis at: `https://[your-username].github.io/[repo-name]/`

## Overview

This project statistically analyzes whether countries with greater economic freedom tend to have better quality-of-life outcomes. It examines correlations between the Heritage Foundation's Economic Freedom Index and multiple quality-of-life metrics.

### Indices Analyzed

| Index | Source | Year |
|-------|--------|------|
| Economic Freedom | Heritage Foundation | 2024 |
| GDP per Capita | World Bank | 2022 |
| Life Expectancy | World Bank | 2022 |
| Infant Mortality | World Bank | 2022 |
| Corruption Perceptions | Transparency International | 2023 |
| Human Development (HDI) | United Nations | 2022 |
| Happiness Score | World Happiness Report | 2024 |
| Peace Score | Global Peace Index | 2024 |
| Democracy Score | Economist Intelligence Unit | 2023 |

## Key Findings

The analysis shows **statistically significant positive correlations** between economic freedom and:
- Lower corruption (r = 0.80)
- Higher democracy scores (r = 0.75)
- Higher GDP per capita (r = 0.72)
- Greater happiness (r = 0.68)
- More peaceful societies (r = 0.67)
- Higher human development (r = 0.63)
- Longer life expectancy (r = 0.58)
- Lower infant mortality (r = -0.51)

**Important Caveat:** Correlation does not imply causation. Many confounding variables exist.

## Project Structure

```
.
├── docs/                          # GitHub Pages website
│   ├── index.html                 # Interactive web interface
│   ├── data.json                  # Cached analysis data
│   └── economic_freedom_analysis.pdf  # PDF report
├── .github/workflows/
│   └── update-data.yml            # GitHub Action for automatic updates
├── economic_freedom_analysis.py    # Main analysis script
├── generate_data.py               # JSON data generator for web
├── requirements.txt               # Python dependencies
└── README.md
```

## Setup

### Prerequisites
- Python 3.9+
- pip

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the analysis:
```bash
# Generate PDF report
python economic_freedom_analysis.py

# Generate web data
python generate_data.py
```

### Deploy to GitHub Pages

1. Push this repository to GitHub

2. Go to repository Settings > Pages

3. Set Source to "Deploy from a branch"

4. Select `main` branch and `/docs` folder

5. Save - your site will be live at `https://[username].github.io/[repo-name]/`

### Automatic Updates

The GitHub Action automatically:
- Runs weekly (every Sunday at midnight UTC)
- Regenerates data when you push to main
- Can be triggered manually via Actions tab

To trigger manually:
1. Go to Actions tab
2. Select "Update Economic Freedom Data"
3. Click "Run workflow"

## How It Works

1. **Data Collection**: The script fetches or uses embedded data from multiple authoritative sources
2. **Data Merging**: Countries are matched across datasets
3. **Statistical Analysis**: Pearson correlations are calculated with p-values
4. **Visualization**: Results are presented in interactive charts and a PDF report

## Methodology

### Correlation Interpretation
- **r > 0.7**: Strong positive correlation
- **r 0.5-0.7**: Moderate positive correlation
- **r 0.3-0.5**: Weak positive correlation
- **r < 0.3**: Very weak or no correlation

### Statistical Significance
- *** p < 0.001 (highly significant)
- ** p < 0.01 (very significant)
- * p < 0.05 (significant)

## Limitations

1. **Correlation ≠ Causation**: These findings show associations, not causal relationships
2. **Confounding Variables**: Historical, geographical, and cultural factors are not controlled
3. **Index Methodology**: Each source has its own biases and limitations
4. **Data Gaps**: Not all countries have complete data for all indices
5. **Snapshot in Time**: Data represents specific years and may change

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - feel free to use this for educational and research purposes.

## Acknowledgments

- Heritage Foundation for the Economic Freedom Index
- World Bank for development indicators
- Transparency International for corruption data
- United Nations for HDI data
- Gallup/WHR for happiness data
- Institute for Economics & Peace for GPI data
- Economist Intelligence Unit for democracy data
