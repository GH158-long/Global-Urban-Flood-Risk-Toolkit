# 📊 Data Classification Codes Reference

This document provides a complete reference for all classification codes used in the Global Urban Flood Risk Analysis Toolkit.

---

## Table of Contents

- [Development Status Codes](#development-status-codes)
- [Income Classification Codes](#income-classification-codes)
- [Continent Codes](#continent-codes)
- [Country Codes](#country-codes)
- [Built-Up Age Categories](#built-up-age-categories)

---

## Development Status Codes

**Column Name:** `Developed`

| Code | Classification | Description |
|:----:|----------------|-------------|
| 1 | Developed | Advanced economies with high industrialization and technological infrastructure |
| 2 | Developing | Emerging and developing economies |

**Usage Example:**
```python
# Filter developed countries
df_developed = df[df['Developed'] == 1]

# Filter developing countries
df_developing = df[df['Developed'] == 2]
```

---

## Income Classification Codes

**Column Name:** `Income_Classification`

Based on World Bank classification (2023 fiscal year):

| Code | Classification | GNI per capita (USD) | Description |
|:----:|----------------|---------------------|-------------|
| 0 | Unclassified | - | Data not available or disputed territories |
| 1 | Low Income | ≤ $1,135 | Least developed economies |
| 2 | Lower Middle Income | $1,136 - $4,465 | Emerging lower-middle economies |
| 3 | Upper Middle Income | $4,466 - $13,845 | Emerging upper-middle economies |
| 4 | High Income | > $13,845 | Advanced economies |

**Examples by Income Level:**

| Level | Example Countries |
|-------|-------------------|
| Low Income | Ethiopia, Somalia, South Sudan, Malawi |
| Lower Middle Income | India, Bangladesh, Kenya, Pakistan |
| Upper Middle Income | China, Brazil, Mexico, Indonesia |
| High Income | USA, Japan, Germany, France |

**Usage Example:**
```python
# Filter high income countries
df_high_income = df[df['Income_Classification'] == 4]

# Filter all middle income countries
df_middle_income = df[df['Income_Classification'].isin([2, 3])]
```

---

## Continent Codes

**Column Name:** `Sovereig`

| Code | Continent | Countries/Territories |
|:----:|-----------|----------------------|
| 10 | Asia | China, India, Japan, South Korea, etc. |
| 11 | Europe | Germany, France, UK, Italy, etc. |
| 12 | Africa | Nigeria, Egypt, South Africa, Kenya, etc. |
| 13 | North America | USA, Canada, Mexico |
| 14 | South America | Brazil, Argentina, Chile, Peru, etc. |
| 15 | Oceania | Australia, New Zealand, Fiji, etc. |
| 17 | Seven Seas | International waters, disputed territories |

**Usage Example:**
```python
# Continent mapping dictionary
CONTINENT_MAP = {
    10: 'Asia',
    11: 'Europe',
    12: 'Africa',
    13: 'North America',
    14: 'South America',
    15: 'Oceania',
    17: 'Seven Seas'
}

# Filter Asian countries
df_asia = df[df['Sovereig'] == 10]
```

---

## Country Codes

**Column Name:** `Country`

### Complete Country Code List (247 entries)

#### Asia (Continent Code: 10)

| Code | Country | Development | Income Level |
|:----:|---------|:-----------:|:------------:|
| 100 | Indonesia | Developing | Upper Middle |
| 101 | Malaysia | Developing | Upper Middle |
| 107 | Cyprus | Developed | High |
| 108 | India | Developing | Lower Middle |
| 109 | China | Developing | Upper Middle |
| 110 | Israel | Developed | High |
| 111 | Palestine | Developed | Lower Middle |
| 112 | Lebanon | Developing | Lower Middle |
| 119 | Syria | Developing | Low |
| 124 | South Korea | Developed | High |
| 125 | North Korea | Developing | Low |
| 132 | Bhutan | Developing | Lower Middle |
| 139 | Oman | Developing | High |
| 140 | Uzbekistan | Developing | Lower Middle |
| 141 | Kazakhstan | Developing | Upper Middle |
| 142 | Tajikistan | Developing | Low |
| 155 | Vietnam | Developing | Lower Middle |
| 156 | Cambodia | Developing | Lower Middle |
| 158 | UAE | Developed | High |
| 160 | Georgia | Developing | Upper Middle |
| 163 | Azerbaijan | Developing | Upper Middle |
| 165 | Turkey | Developing | Upper Middle |
| 167 | Laos | Developing | Lower Middle |
| 168 | Kyrgyzstan | Developing | Lower Middle |
| 169 | Armenia | Developing | Upper Middle |
| 189 | Iraq | Developing | Upper Middle |
| 192 | Iran | Developing | Lower Middle |
| 204 | Qatar | Developed | High |
| 205 | Saudi Arabia | Developing | High |
| 208 | Pakistan | Developing | Lower Middle |
| 210 | Thailand | Developing | Upper Middle |
| 218 | Timor-Leste | Developing | Lower Middle |
| 219 | Brunei | Developing | High |
| 226 | Myanmar | Developing | Lower Middle |
| 227 | Bangladesh | Developing | Lower Middle |
| 229 | Afghanistan | Developing | Low |
| 242 | Turkmenistan | Developing | Upper Middle |
| 243 | Jordan | Developing | Upper Middle |
| 244 | Nepal | Developing | Lower Middle |
| 262 | Yemen | Developing | Low |
| 266 | Hong Kong | Developed | High |
| 282 | Philippines | Developing | Lower Middle |
| 283 | Sri Lanka | Developing | Lower Middle |
| 288 | Taiwan | Developed | High |
| 289 | Japan | Developed | High |
| 330 | Singapore | Developed | High |
| 338 | Maldives | Developing | Upper Middle |
| 349 | Bahrain | Developed | High |
| 353 | Macau | Developed | High |

#### Europe (Continent Code: 11)

| Code | Country | Development | Income Level |
|:----:|---------|:-----------:|:------------:|
| 121 | France | Developed | High |
| 133 | Ukraine | Developing | Lower Middle |
| 134 | Belarus | Developing | Upper Middle |
| 143 | Lithuania | Developed | High |
| 147 | Russia | Developing | Upper Middle |
| 148 | Czechia | Developed | High |
| 149 | Germany | Developed | High |
| 150 | Estonia | Developed | High |
| 151 | Latvia | Developed | High |
| 152 | Norway | Developed | High |
| 153 | Sweden | Developed | High |
| 154 | Finland | Developed | High |
| 157 | Luxembourg | Developed | High |
| 159 | Belgium | Developed | High |
| 161 | North Macedonia | Developing | Upper Middle |
| 162 | Albania | Developing | Upper Middle |
| 164 | Kosovo | Developing | Upper Middle |
| 166 | Spain | Developed | High |
| 170 | Denmark | Developed | High |
| 173 | Romania | Developed | High |
| 174 | Hungary | Developed | High |
| 175 | Slovakia | Developed | High |
| 176 | Poland | Developed | High |
| 177 | Ireland | Developed | High |
| 178 | United Kingdom | Developed | High |
| 179 | Greece | Developed | High |
| 188 | Austria | Developed | High |
| 190 | Italy | Developed | High |
| 191 | Switzerland | Developed | High |
| 193 | Netherlands | Developed | High |
| 194 | Liechtenstein | Developed | High |
| 196 | Serbia | Developing | Upper Middle |
| 202 | Croatia | Developed | High |
| 203 | Slovenia | Developed | High |
| 209 | Bulgaria | Developed | Upper Middle |
| 211 | San Marino | Developed | High |
| 220 | Monaco | Developed | High |
| 228 | Andorra | Developed | High |
| 230 | Montenegro | Developing | Upper Middle |
| 231 | Bosnia & Herzegovina | Developing | Upper Middle |
| 240 | Portugal | Developed | High |
| 241 | Moldova | Developing | Upper Middle |
| 253 | Gibraltar | Developed | High |
| 267 | Vatican City | Developed | High |
| 291 | Iceland | Developed | High |
| 322 | Malta | Developed | High |
| 323 | Jersey | Developed | High |
| 324 | Guernsey | Developed | High |
| 325 | Isle of Man | Developed | High |
| 327 | Faroe Islands | Developed | High |

#### Africa (Continent Code: 12)

| Code | Country | Development | Income Level |
|:----:|---------|:-----------:|:------------:|
| 113 | Ethiopia | Developing | Low |
| 114 | South Sudan | Developing | Low |
| 115 | Somalia | Developing | Low |
| 116 | Kenya | Developing | Lower Middle |
| 117 | Malawi | Developing | Low |
| 118 | Tanzania | Developing | Lower Middle |
| 120 | Somaliland | Developing | Low |
| 126 | Morocco | Developing | Lower Middle |
| 127 | Western Sahara | Developing | Lower Middle |
| 130 | Congo | Developing | Lower Middle |
| 131 | DR Congo | Developing | Low |
| 135 | Namibia | Developing | Upper Middle |
| 136 | South Africa | Developing | Upper Middle |
| 171 | Libya | Developing | Upper Middle |
| 172 | Tunisia | Developing | Lower Middle |
| 180 | Zambia | Developing | Lower Middle |
| 181 | Sierra Leone | Developing | Low |
| 182 | Guinea | Developing | Low |
| 183 | Liberia | Developing | Low |
| 184 | CAR | Developing | Low |
| 185 | Sudan | Developing | Low |
| 186 | Djibouti | Developing | Lower Middle |
| 187 | Eritrea | Developing | Low |
| 195 | Cote d'Ivoire | Developing | Lower Middle |
| 197 | Mali | Developing | Low |
| 198 | Senegal | Developing | Lower Middle |
| 199 | Nigeria | Developing | Lower Middle |
| 200 | Benin | Developing | Lower Middle |
| 201 | Angola | Developing | Lower Middle |
| 206 | Botswana | Developing | Upper Middle |
| 207 | Zimbabwe | Developing | Lower Middle |
| 214 | Chad | Developing | Low |
| 221 | Algeria | Developing | Lower Middle |
| 222 | Mozambique | Developing | Low |
| 223 | Eswatini | Developing | Lower Middle |
| 224 | Burundi | Developing | Low |
| 225 | Rwanda | Developing | Low |
| 232 | Uganda | Developing | Low |
| 245 | Lesotho | Developing | Lower Middle |
| 246 | Cameroon | Developing | Lower Middle |
| 247 | Gabon | Developing | Upper Middle |
| 248 | Niger | Developing | Low |
| 249 | Burkina Faso | Developing | Low |
| 250 | Togo | Developing | Low |
| 251 | Ghana | Developing | Lower Middle |
| 252 | Guinea-Bissau | Developing | Low |
| 261 | Egypt | Developing | Lower Middle |
| 263 | Mauritania | Developing | Lower Middle |
| 264 | Equatorial Guinea | Developing | Upper Middle |
| 265 | Gambia | Developing | Low |
| 281 | Madagascar | Developing | Low |
| 295 | Seychelles | Developing | High |
| 317 | Saint Helena | Developing | Upper Middle |
| 318 | Mauritius | Developing | Upper Middle |
| 319 | Comoros | Developing | Lower Middle |
| 320 | Sao Tome & Principe | Developing | Lower Middle |
| 321 | Cape Verde | Developing | Lower Middle |

#### North America (Continent Code: 13)

| Code | Country | Development | Income Level |
|:----:|---------|:-----------:|:------------:|
| 128 | Costa Rica | Developing | Upper Middle |
| 129 | Nicaragua | Developing | Lower Middle |
| 212 | Haiti | Developing | Low |
| 213 | Dominican Rep. | Developing | Upper Middle |
| 216 | El Salvador | Developing | Lower Middle |
| 217 | Guatemala | Developing | Upper Middle |
| 233 | Guantanamo Bay | Developed | High |
| 234 | Cuba | Developing | Upper Middle |
| 235 | Honduras | Developing | Lower Middle |
| 254 | USA | Developed | High |
| 255 | Canada | Developed | High |
| 256 | Mexico | Developing | Upper Middle |
| 257 | Belize | Developing | Upper Middle |
| 258 | Panama | Developing | High |
| 277 | Greenland | Developed | High |
| 286 | Bahamas | Developing | High |
| 287 | Turks & Caicos | Developing | High |
| 290 | St Pierre & Miquelon | Developed | High |
| 298 | Trinidad & Tobago | Developing | High |
| 299 | Grenada | Developing | Upper Middle |
| 300 | St Vincent | Developing | Upper Middle |
| 301 | Barbados | Developing | High |
| 302 | Saint Lucia | Developing | Upper Middle |
| 303 | Dominica | Developing | Upper Middle |
| 305 | Montserrat | Developing | Upper Middle |
| 306 | Antigua & Barbuda | Developing | High |
| 307 | St Kitts & Nevis | Developing | High |
| 308 | US Virgin Islands | Developed | High |
| 309 | St Barthelemy | Developed | High |
| 310 | Puerto Rico | Developed | High |
| 311 | Anguilla | Developing | High |
| 312 | British Virgin Is. | Developing | High |
| 313 | Jamaica | Developing | Upper Middle |
| 314 | Cayman Islands | Developing | High |
| 315 | Bermuda | Developing | High |

#### South America (Continent Code: 14)

| Code | Country | Development | Income Level |
|:----:|---------|:-----------:|:------------:|
| 102 | Chile | Developing | High |
| 103 | Bolivia | Developing | Lower Middle |
| 104 | Peru | Developing | Upper Middle |
| 105 | Argentina | Developing | Upper Middle |
| 122 | Suriname | Developing | Upper Middle |
| 123 | Guyana | Developing | Upper Middle |
| 144 | Brazil | Developing | Upper Middle |
| 145 | Uruguay | Developing | High |
| 236 | Ecuador | Developing | Upper Middle |
| 237 | Colombia | Developing | Upper Middle |
| 238 | Paraguay | Developing | Upper Middle |
| 259 | Venezuela | Developing | Upper Middle |
| 284 | Curacao | Developing | High |
| 285 | Aruba | Developing | High |
| 342 | Falkland Islands | Developed | High |

#### Oceania (Continent Code: 15)

| Code | Country | Development | Income Level |
|:----:|---------|:-----------:|:------------:|
| 260 | Papua New Guinea | Developing | Lower Middle |
| 276 | Australia | Developed | High |
| 278 | Fiji | Developing | Upper Middle |
| 279 | New Zealand | Developed | High |
| 280 | New Caledonia | Developed | High |
| 292 | Pitcairn Islands | Developing | Upper Middle |
| 293 | French Polynesia | Developed | High |
| 296 | Kiribati | Developing | Lower Middle |
| 297 | Marshall Islands | Developing | Upper Middle |
| 331 | Norfolk Island | Developed | High |
| 332 | Cook Islands | Developing | Upper Middle |
| 333 | Tonga | Developing | Upper Middle |
| 334 | Wallis & Futuna | Developing | Upper Middle |
| 335 | Samoa | Developing | Lower Middle |
| 336 | Solomon Islands | Developing | Lower Middle |
| 337 | Tuvalu | Developing | Upper Middle |
| 339 | Nauru | Developing | High |
| 340 | Micronesia | Developing | Lower Middle |
| 343 | Vanuatu | Developing | Lower Middle |
| 344 | Niue NZ | Developing | Upper Middle |
| 345 | American Samoa | Developing | Upper Middle |
| 346 | Palau | Developing | High |
| 347 | Guam | Developed | High |
| 348 | N. Mariana Islands | Developing | High |

---

## Built-Up Age Categories

**Column Name:** `Built_Up_Age`

Represents when urban areas were first developed, in 5-year intervals from 1975-2025:

| Code | Period | Age Range | Description |
|:----:|--------|-----------|-------------|
| 5 | 2020-2025 | 0-5 years | Newest development |
| 10 | 2015-2020 | 5-10 years | Recent development |
| 15 | 2010-2015 | 10-15 years | - |
| 20 | 2005-2010 | 15-20 years | - |
| 25 | 2000-2005 | 20-25 years | - |
| 30 | 1995-2000 | 25-30 years | - |
| 35 | 1990-1995 | 30-35 years | - |
| 40 | 1985-1990 | 35-40 years | - |
| 45 | 1980-1985 | 40-45 years | - |
| 50 | 1975-1980 | 45-50 years | Oldest in dataset |

**Research Significance:**

- **Newer areas (5-15):** Modern construction standards, potentially better flood resilience
- **Middle-aged areas (20-35):** Mix of standards, may need retrofitting
- **Older areas (40-50):** Legacy infrastructure, higher vulnerability potential

**Usage Example:**
```python
# Filter newest development
df_new = df[df['Built_Up_Age'] == 5]

# Filter development older than 30 years
df_old = df[df['Built_Up_Age'] >= 35]
```

---

## Data Quality Notes

1. **Code 0 (Unclassified):** Used for areas with insufficient data or disputed territories
2. **Missing Values:** Some records may have `NaN` values; handle appropriately in analysis
3. **Temporal Reference:** Income classifications reflect 2023 World Bank data
4. **Sovereignty:** Some territories have disputed sovereignty (e.g., Palestine, Taiwan)

---

## Python Code Reference

```python
# Complete mapping dictionaries from the toolkit

DEVELOPED_MAP = {
    1: 'Developed',
    2: 'Developing'
}

INCOME_MAP = {
    0: 'Unclassified',
    1: 'Low Income',
    2: 'Lower Middle Income',
    3: 'Upper Middle Income',
    4: 'High Income'
}

CONTINENT_MAP = {
    10: 'Asia',
    11: 'Europe',
    12: 'Africa',
    13: 'North America',
    14: 'South America',
    15: 'Oceania',
    17: 'Seven Seas'
}

AGE_CATEGORIES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
```

---

## Related Files

- `Countries2_classified.xlsx` - Full country classification reference table
- `README.md` - Main documentation
- Source code files contain embedded mapping dictionaries
