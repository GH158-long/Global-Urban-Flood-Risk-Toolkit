# 🌊 Global Urban Flood Risk Analysis Toolkit

A comprehensive Python toolkit for analyzing global urban flood risk, calculating Expected Annual Damage (EAD), and performing statistical analysis of built-up area age-risk relationships.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Tool 1: Parquet Field Calculator Pro](#tool-1-parquet-field-calculator-pro)
- [Tool 2: Flood Hazard EAD Calculator](#tool-2-flood-hazard-ead-calculator)
- [Tool 3: Urban Flood Comprehensive Analyzer](#tool-3-urban-flood-comprehensive-analyzer)
- [Data Classification Codes](#data-classification-codes)
- [References](#references)

---

## Overview

This toolkit consists of three integrated GUI applications designed for large-scale urban flood risk analysis:

| Tool | Purpose | Key Features |
|------|---------|--------------|
| **Parquet Field Calculator Pro** | Data manipulation and field calculation | Expression-based calculations, field management |
| **Flood Hazard EAD Calculator** | EAD and Risk Index computation | Trapezoidal integration, chunked processing |
| **Urban Flood Comprehensive Analyzer** | Statistical analysis and reporting | Multi-dimensional statistics, Excel export |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- 16GB+ RAM recommended (64GB for large datasets)
- Windows 10/11 (optimized for Windows, may work on Linux/Mac with modifications)

### Dependencies

```bash
pip install -r requirements.txt
```

---

## Tool 1: Parquet Field Calculator Pro

### Description

A powerful GUI application for manipulating large Parquet files (30GB+). Supports custom field calculations using mathematical expressions, field management (add, delete, rename, reorder), and efficient chunked processing.

### Features

- ✅ Load and analyze Parquet files with lazy evaluation (Polars)
- ✅ Create new fields using mathematical expressions
- ✅ Support for 15+ mathematical functions
- ✅ Field management: add, delete, rename, reorder
- ✅ Real-time expression validation
- ✅ Progress tracking and logging
- ✅ Multi-threaded processing

### Supported Mathematical Functions

| Function | Description | Example |
|----------|-------------|---------|
| `abs(x)` | Absolute value | `abs([field])` |
| `sqrt(x)` | Square root | `sqrt([field])` |
| `log(x)` | Natural logarithm | `log([field])` |
| `log10(x)` | Base-10 logarithm | `log10([field])` |
| `exp(x)` | Exponential | `exp([field])` |
| `sin(x)`, `cos(x)`, `tan(x)` | Trigonometric | `sin([angle])` |
| `ceil(x)` | Round up | `ceil([field])` |
| `floor(x)` | Round down | `floor([field])` |
| `round(x)` | Round to nearest integer | `round([field])` |
| `sign(x)` | Sign function (-1, 0, 1) | `sign([field])` |
| `min(a, b)` | Minimum of two values | `min([field1], [field2])` |
| `max(a, b)` | Maximum of two values | `max([field1], [field2])` |

### Expression Syntax

Field references use square brackets: `[field_name]`

**Operators:**
- Arithmetic: `+`, `-`, `*`, `/`
- Power: `^` (converts to `**`)
- Parentheses: `(`, `)`

### Calculation Examples

#### Example 1: Normalized Hazard Index

Create a normalized hazard index from raw hazard values:

```
# Expression
[Hazard_Index_RP100] / max([Hazard_Index_RP100], 0.001)

# Result Field Name
Hazard_Normalized
```

#### Example 2: Combined Risk Score

Calculate a composite risk score:

```
# Expression
([Hazard_Index_EAD] * 0.4) + ([Exposure_Index] * 0.3) + ([Vulnerability_Index] * 0.3)

# Result Field Name
Risk_Weighted
```

#### Example 3: Distance-Based Decay Function

Apply exponential decay based on distance from river:

```
# Expression
exp(-[Distance_from_River] / 1000)

# Result Field Name
River_Proximity_Factor
```

#### Example 4: Slope Correction Factor

Calculate terrain-adjusted values:

```
# Expression
[Hazard_Index_EAD] * (1 + sin([Slope] * 3.14159 / 180))

# Result Field Name
Slope_Adjusted_Hazard
```

### Usage

```bash
python Parquet_Field_Calculator.py
```

1. Click **📂 Browse** to select input Parquet file
2. Click **▶ Load & Analyze** to scan the file
3. Enter expression in the calculator area
4. Enter output field name
5. Click **🚀 Execute** to calculate
6. Click **💾 Save** to export results

---

## Tool 2: Flood Hazard EAD Calculator

### Description

Calculates Expected Annual Damage (EAD) using trapezoidal integration across multiple return periods, and computes a composite Flood Risk Index based on the IPCC framework.

### Core Calculations

#### Expected Annual Damage (EAD)

The EAD represents the average annual flood damage expected at a location, calculated by integrating flood depth over the probability of occurrence.

**Mathematical Formula:**

```
EAD_i = ∫₀¹ H_i(p) dp
```

Where:
- `p` is the Annual Exceedance Probability (AEP = 1/T)
- `H_i(p)` is the normalized flood hazard index at grid cell `i`
- `T` is the return period in years

**Trapezoidal Integration Method:**

```
EAD = Σ [(H_{j} + H_{j+1}) / 2] × Δp_j
```

Where:
- `H_j` is the hazard value at probability point `j`
- `Δp_j = p_{j+1} - p_j` is the probability interval

**Step-by-Step Calculation Example:**

Given hazard values for return periods [10, 20, 50, 75, 100, 200, 500] years:

| Return Period (T) | Hazard (H) | AEP (p = 1/T) |
|-------------------|------------|---------------|
| 10 years | 0.20 | 0.1000 |
| 20 years | 0.35 | 0.0500 |
| 50 years | 0.50 | 0.0200 |
| 75 years | 0.58 | 0.0133 |
| 100 years | 0.65 | 0.0100 |
| 200 years | 0.78 | 0.0050 |
| 500 years | 0.92 | 0.0020 |

**Step 1:** Sort by probability (ascending): p = [0.002, 0.005, 0.01, 0.0133, 0.02, 0.05, 0.1]

**Step 2:** Add boundary conditions:
- p = 0: Use H from 500-year RP (0.92)
- p = 1: Use H from 10-year RP (0.20)

Extended arrays:
```
p_ext = [0, 0.002, 0.005, 0.01, 0.0133, 0.02, 0.05, 0.1, 1]
H_ext = [0.92, 0.92, 0.78, 0.65, 0.58, 0.50, 0.35, 0.20, 0.20]
```

**Step 3:** Apply trapezoidal rule:

```
EAD = (0.92+0.92)/2 × 0.002 + (0.92+0.78)/2 × 0.003 + (0.78+0.65)/2 × 0.005 + 
      (0.65+0.58)/2 × 0.0033 + (0.58+0.50)/2 × 0.0067 + (0.50+0.35)/2 × 0.03 + 
      (0.35+0.20)/2 × 0.05 + (0.20+0.20)/2 × 0.9

EAD = 0.00184 + 0.00255 + 0.00358 + 0.00203 + 0.00362 + 0.01275 + 0.01375 + 0.18
EAD = 0.2201
```

**Step 4:** Normalize by EAD_max (when all H=1):
```
EAD_normalized = 0.2201 / EAD_max = 0.2201 / 1.0 = 0.2201
```

#### Composite Risk Index

Calculated using the geometric mean formula from the IPCC framework:

**Formula:**

```
Risk_i = (Hazard_i × Exposure_i × Vulnerability_i)^(1/3)
```

**Calculation Example:**

Given:
- Hazard_Index_EAD = 0.65
- Exposure_Index = 0.80
- Vulnerability_Index = 0.45

```
Risk = (0.65 × 0.80 × 0.45)^(1/3)
Risk = (0.234)^(1/3)
Risk = 0.616
```

### Input Requirements

| Column Name | Description | Value Range |
|-------------|-------------|-------------|
| `Hazard_Index_RP010` | 10-year return period hazard | 0.0 - 1.0 |
| `Hazard_Index_RP020` | 20-year return period hazard | 0.0 - 1.0 |
| `Hazard_Index_RP050` | 50-year return period hazard | 0.0 - 1.0 |
| `Hazard_Index_RP075` | 75-year return period hazard | 0.0 - 1.0 |
| `Hazard_Index_RP100` | 100-year return period hazard | 0.0 - 1.0 |
| `Hazard_Index_RP200` | 200-year return period hazard | 0.0 - 1.0 |
| `Hazard_Index_RP500` | 500-year return period hazard | 0.0 - 1.0 |
| `Exposure_Index` | Population/asset exposure | 0.0 - 1.0 |
| `Vulnerability_Index` | Structural vulnerability | 0.0 - 1.0 |

### Output Fields

| Column Name | Description |
|-------------|-------------|
| `Hazard_Index_EAD` | Normalized Expected Annual Damage (0-1) |
| `Risk_Index` | Composite Flood Risk Index (0-1) |

### Usage

```bash
python Calculate_the_EAD_and_flood_risk_index.py
```

1. Select input Parquet file
2. Set output directory and filename
3. Adjust chunk size if needed (default: 500,000 rows)
4. Click **Validate** to check input file
5. Click **Start Processing** to compute

---

## Tool 3: Urban Flood Comprehensive Analyzer

### Description

Performs comprehensive statistical analysis of urban flood risk across multiple dimensions (development status, income level, continent, country) and generates detailed Excel reports.

### Analysis Dimensions

#### Built-Up Age Categories
Represents the age of urban built-up areas in 5-year intervals:

| Code | Description |
|------|-------------|
| 5 | 0-5 years old (2020-2025) |
| 10 | 5-10 years old (2015-2020) |
| 15 | 10-15 years old (2010-2015) |
| 20 | 15-20 years old (2005-2010) |
| 25 | 20-25 years old (2000-2005) |
| 30 | 25-30 years old (1995-2000) |
| 35 | 30-35 years old (1990-1995) |
| 40 | 35-40 years old (1985-1990) |
| 45 | 40-45 years old (1980-1985) |
| 50 | 45-50 years old (1975-1980) |

### Statistical Computations

#### 1. Area Statistics

Calculates total grid count and area (km²) by development status and age:

```
Area (km²) = Grid_Count × 0.01
```

(Each grid cell = 100m × 100m = 0.01 km²)

#### 2. Risk Index Statistics

For each grouping (Development Status, Income, Continent, Country):

| Statistic | Formula | Description |
|-----------|---------|-------------|
| Mean (μ) | `μ = Σx / n` | Average risk value |
| Std Dev (σ) | `σ = √[Σ(x-μ)² / n]` | Standard deviation |
| Standard Error (SE) | `SE = σ / √n` | Standard error of the mean |
| Min | `min(x)` | Minimum value |
| Max | `max(x)` | Maximum value |
| Count (n) | `count(x)` | Sample size |

**Calculation Example:**

For a country with Risk_Index values: [0.45, 0.52, 0.48, 0.55, 0.50]

```
n = 5
Mean = (0.45 + 0.52 + 0.48 + 0.55 + 0.50) / 5 = 0.50
Variance = [(0.45-0.50)² + (0.52-0.50)² + (0.48-0.50)² + (0.55-0.50)² + (0.50-0.50)²] / 5
         = [0.0025 + 0.0004 + 0.0004 + 0.0025 + 0] / 5 = 0.00116
Std Dev = √0.00116 = 0.034
SE = 0.034 / √5 = 0.015
```

#### 3. Environmental Factor Statistics

Computes statistics for environmental factors:

| Factor | Description | Unit |
|--------|-------------|------|
| `Hazard_Index_EAD` | Expected Annual Damage | 0-1 |
| `Slope` | Terrain slope | Degrees |
| `DEM` | Digital Elevation Model | Meters |
| `Distance_from_River` | River proximity | Meters |

#### 4. Population Exposure Analysis

**Exposure Ratio:**

```
Exposure_Ratio (%) = (Population_Exposed / Total_Population) × 100
```

**Time Series Analysis:**
- Years analyzed: 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025
- Grouped by country and income level

### Output Excel Report Structure

| Sheet | Content |
|-------|---------|
| Overview | Summary statistics, metadata |
| Area_Statistics | Grid count and area by development status × age |
| Risk_by_Development | Risk statistics grouped by Developed/Developing |
| Risk_by_Income | Risk statistics grouped by income level |
| Risk_by_Continent | Risk statistics grouped by continent |
| Risk_by_Country | Risk statistics grouped by country |
| Env_by_Development | Environmental factors by development status |
| Env_by_Income | Environmental factors by income level |
| Env_by_Continent | Environmental factors by continent |
| Env_by_Country | Environmental factors by country |
| Exposure_Global | Global population exposure ratios |
| Exposure_by_Development | Exposure by development status |
| Exposure_by_Income | Exposure by income level |
| Exposure_by_Continent | Exposure by continent |
| Exposure_by_Country | Exposure by country |
| Exposure_Yearly | Time series exposure data |

### Usage

```bash
python Statistical_Analysis_of_Age-Risk_Relationship.py
```

1. Select input Parquet file
2. Set output directory
3. Toggle "Exclude Zero Values" as needed
4. Click **▶ Start Analysis**

---

## Data Classification Codes

### Development Status (`Developed` column)

| Code | Classification |
|------|----------------|
| 1 | Developed |
| 2 | Developing |

### Income Classification (`Income_Classification` column)

| Code | Classification | World Bank Definition |
|------|----------------|----------------------|
| 0 | Unclassified | - |
| 1 | Low Income | GNI ≤ $1,135 |
| 2 | Lower Middle Income | $1,136 - $4,465 |
| 3 | Upper Middle Income | $4,466 - $13,845 |
| 4 | High Income | GNI > $13,845 |

### Continent Codes (`Sovereig` column)

| Code | Continent |
|------|-----------|
| 10 | Asia |
| 11 | Europe |
| 12 | Africa |
| 13 | North America |
| 14 | South America |
| 15 | Oceania |
| 17 | Seven Seas (Disputed/International) |

### Country Codes (`Country` column)

Complete country code mapping (247 countries/territories):

<details>
<summary>Click to expand full country code list</summary>

| Code | Country | Code | Country |
|------|---------|------|---------|
| 100 | Indonesia | 200 | Benin |
| 101 | Malaysia | 201 | Angola |
| 102 | Chile | 202 | Croatia |
| 103 | Bolivia | 203 | Slovenia |
| 104 | Peru | 204 | Qatar |
| 105 | Argentina | 205 | Saudi Arabia |
| 106 | Zekeliya Barracks | 206 | Botswana |
| 107 | Cyprus | 207 | Zimbabwe |
| 108 | India | 208 | Pakistan |
| 109 | China | 209 | Bulgaria |
| 110 | Israel | 210 | Thailand |
| 111 | Palestine | 211 | San Marino |
| 112 | Lebanon | 212 | Haiti |
| 113 | Ethiopia | 213 | Dominican Rep. |
| 114 | South Sudan | 214 | Chad |
| 115 | Somalia | 215 | Kuwait |
| 116 | Kenya | 216 | El Salvador |
| 117 | Malawi | 217 | Guatemala |
| 118 | Tanzania | 218 | Timor-Leste |
| 119 | Syria | 219 | Brunei |
| 120 | Somaliland | 220 | Monaco |
| 121 | France | 221 | Algeria |
| 122 | Suriname | 222 | Mozambique |
| 123 | Guyana | 223 | Eswatini |
| 124 | South Korea | 224 | Burundi |
| 125 | North Korea | 225 | Rwanda |
| 126 | Morocco | 226 | Myanmar |
| 127 | Western Sahara | 227 | Bangladesh |
| 128 | Costa Rica | 228 | Andorra |
| 129 | Nicaragua | 229 | Afghanistan |
| 130 | Congo | 230 | Montenegro |
| 131 | DR Congo | 231 | Bosnia & Herzegovina |
| 132 | Bhutan | 232 | Uganda |
| 133 | Ukraine | 233 | Guantanamo Bay |
| 134 | Belarus | 234 | Cuba |
| 135 | Namibia | 235 | Honduras |
| 136 | South Africa | 236 | Ecuador |
| 137 | Saint Martin | 237 | Colombia |
| 138 | Sint Maarten | 238 | Paraguay |
| 139 | Oman | 239 | Brazil Island |
| 140 | Uzbekistan | 240 | Portugal |
| 141 | Kazakhstan | 241 | Moldova |
| 142 | Tajikistan | 242 | Turkmenistan |
| 143 | Lithuania | 243 | Jordan |
| 144 | Brazil | 244 | Nepal |
| 145 | Uruguay | 245 | Lesotho |
| 146 | Mongolia | 246 | Cameroon |
| 147 | Russia | 247 | Gabon |
| 148 | Czechia | 248 | Niger |
| 149 | Germany | 249 | Burkina Faso |
| 150 | Estonia | 250 | Togo |
| 151 | Latvia | 251 | Ghana |
| 152 | Norway | 252 | Guinea-Bissau |
| 153 | Sweden | 253 | Gibraltar |
| 154 | Finland | 254 | USA |
| 155 | Vietnam | 255 | Canada |
| 156 | Cambodia | 256 | Mexico |
| 157 | Luxembourg | 257 | Belize |
| 158 | UAE | 258 | Panama |
| 159 | Belgium | 259 | Venezuela |
| 160 | Georgia | 260 | Papua New Guinea |
| 161 | North Macedonia | 261 | Egypt |
| 162 | Albania | 262 | Yemen |
| 163 | Azerbaijan | 263 | Mauritania |
| 164 | Kosovo | 264 | Equatorial Guinea |
| 165 | Turkey | 265 | Gambia |
| 166 | Spain | 266 | Hong Kong |
| 167 | Laos | 267 | Vatican City |
| 168 | Kyrgyzstan | 268 | N. Cyprus |
| 169 | Armenia | 269 | UN Buffer Zone |
| 170 | Denmark | 270 | Siachen Glacier |
| 171 | Libya | 271 | Baikonur |
| 172 | Tunisia | 272 | Akrotiri SBA |
| 173 | Romania | 273 | S. Patagonia Ice Field |
| 174 | Hungary | 274 | Bir Tawil |
| 175 | Slovakia | 276 | Australia |
| 176 | Poland | 277 | Greenland |
| 177 | Ireland | 278 | Fiji |
| 178 | United Kingdom | 279 | New Zealand |
| 179 | Greece | 280 | New Caledonia |
| 180 | Zambia | 281 | Madagascar |
| 181 | Sierra Leone | 282 | Philippines |
| 182 | Guinea | 283 | Sri Lanka |
| 183 | Liberia | 284 | Curacao |
| 184 | CAR | 285 | Aruba |
| 185 | Sudan | 286 | Bahamas |
| 186 | Djibouti | 287 | Turks & Caicos |
| 187 | Eritrea | 288 | Taiwan |
| 188 | Austria | 289 | Japan |
| 189 | Iraq | 290 | St Pierre & Miquelon |
| 190 | Italy | 291 | Iceland |
| 191 | Switzerland | 292 | Pitcairn Islands |
| 192 | Iran | 293 | French Polynesia |
| 193 | Netherlands | 294 | TAAF France |
| 194 | Liechtenstein | 295 | Seychelles |
| 195 | Cote d'Ivoire | 296 | Kiribati |
| 196 | Serbia | 297 | Marshall Islands |
| 197 | Mali | 298 | Trinidad & Tobago |
| 198 | Senegal | 299 | Grenada |
| 199 | Nigeria | ... | ... |

</details>

---

## References

1. Meyer, V., et al. (2009). Review article: Assessing integrated environmental models. *IEAM*, 5(1), 17-26.
2. De Moel, H., et al. (2015). Flood risk assessments at different spatial scales. *MASGC*, 20(6), 865-890.
3. Wing, O.E.J., et al. (2022). Inequitable patterns of US flood risk. *Nature Climate Change*, 12, 156-162.
4. IPCC (2014). Climate Change 2014: Impacts, Adaptation, and Vulnerability. Cambridge University Press.
5. World Bank (2023). World Bank Country and Lending Groups. https://datahelpdesk.worldbank.org/

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

**LONG** - 2026

For questions or issues, please open a GitHub issue.
