
# Geocoding Project - Toponym Resolution

## Overview
This repository contains the code for resolving toponyms (place names) by matching them to geographical locations using hierarchical paths and AI-assisted methods.

## Setup
Dependencies
Install required Python packages:
```bash
pip install pandas python-Levenshtein openai ged4py
```
Required Python libraries:
- Python 3.8+
- pandas - Data manipulation
- python-Levenshtein - String similarity calculations
- openai - AI-assisted place disambiguation (optional)
- ged4py - GEDCOM file parsing
- Standard libraries: os, sys, re, csv, zipfile, datetime, collections

### Directory Structure
```
geocoding_project/
├── data/
│   ├── task/                      # Sample GEDCOM files
│   └── AI-review/                 # Places that resolved by AI and user review
│       ├── input/                 # Unmatched places for AI
│       ├── batches/               # processed batches
│       ├── output/                # AI JSON and CSV outputs
│       └── output_review/         # Human-reviewed AI results
│       └── user/                  # User manual review results
├── counties_references/           # County reference CSVs by year
│   ├── 1790.csv
│   ├── 1800.csv
│   └── ...
├── city_references/              # City reference CSVs by year
│   ├── 1790.csv
│   ├── 1800.csv
│   └── ...
├── output/                       # Matched place results summaries
└── validation/                   # human-AI output comparison                      
```

## Key Steps
### 1. Prepare Reference Data

**Download Reference Data:**
- County-level GIS files (1790-1940): https://www.nhgis.org/
- City/township-level GIS files (1790-1940): https://www.openicpsr.org/openicpsr/project/179401/version/V2/view

**Convert to CSV Format:**
Convert downloaded shapefiles to CSV with the following structure:

**Counties Reference** (`counties_references/YEAR.csv`):
```csv
state,county
pennsylvania,philadelphia
pennsylvania,lancaster
new york,kings
```

**Cities Reference** (`city_references/YEAR.csv`):
```csv
state,county,stdtownship
pennsylvania,philadelphia,philadelphia
pennsylvania,lancaster,lancaster
new york,kings,brooklyn
```

Create one CSV file for each census year (1790, 1800, 1810, 1820, ..., 1940).


### 2. Parse GEDCOM Files and Match Places (Main Step)

**Toponym Resolution:**

The main processing script parses GEDCOM genealogical files and matches birthplaces to geographic locations using hierarchical path matching.

**Script:** `Deterministic_match.py`
**Features:**
- Processes zipped GEDCOM files
- Standardizes place names (removes punctuation, handles abbreviations)
- Matches places hierarchically (state → county → city)
- Filters foreign-born individuals
- Loads year-appropriate reference data based on birth year
- Outputs results to multiple CSV files by match quality

**Input:**
- Zipped GEDCOM files
- Year-based reference CSVs
**Output CSV Files:**
- `persons.csv` - Individual records with matched places
- `city_county_state.csv` - Complete matches (city + county + state)
- `county_state.csv` - County and state matches
- `city_and_state_nocounty.csv` - City and state matches (county not found)
- `county_and_state_nocity.csv` - County and state matches (city not found)
- `state_nocounty_nocity.csv` - State-only matches
- `nostate_nocounty_nocity.csv` - No matches (requires manual review)
- `processing_stats.txt` - Summary statistics

**Usage:**
```python
process_gedcom_files(
    zip_path="data/task/sample_gedcoms.zip",
    counties_dir="counties_references",
    cities_dir="city_references",
    output_dir="output_results"
)
```
### 3. Handle Special Cases: Colorado "co co"

**Colorado Disambiguation:**

Birthplaces ending with "co co" are ambiguous (could be "Colorado" or "County County"). This script assumes Colorado and performs multi-year searches.

**Script:** `Co_special_case.py`

**Features:**
- Identifies birthplaces ending with " co co"
- Searches across multiple census years (up to 14 iterations)
- Uses progressive year-jumping strategy
- Separates first-search vs second-search results

**Input:**
- `nostate_nocounty_nocity.csv` (filtered for "co co" endings)

**Output:**
- Same structure as main matching, plus:
  - `second_search_city_county_state.csv`
  - `second_search_county_and_state_nocity.csv`
  - `second_search_city_and_state_nocounty.csv`
  - `colorado_processing_stats.txt`

**Usage:**
```python
process_colorado_birthplaces(
    input_file="output_results/nostate_nocounty_nocity.csv",
    counties_dir="counties_references",
    cities_dir="city_references",
    output_dir="output_results/colorado_results"
)
```
### 4. AI-Assisted Toponym Disambiguation

**Using AI for Unmatched Places:**

For places that couldn't be matched automatically, use AI (GPT-4o) to suggest state, county, and city.

**Scripts:**
- `LLM_recognition.py` - Sends unmatched places to OpenAI API (requires API key)
- `Reading_JSON.py` - Processes AI responses into CSV format

**Input:**
- `state_nocounty_nocity.csv`
- `nostate_nocounty_nocity.csv`

**Processing:**
1. Batch unmatched places
2. Send to OpenAI API with system prompts
3. Receive JSON responses with suggested matches
4. Convert to CSV for human review

**Output:**
- `AI_review/output`

### 5. Human Review and Validation

**Manual Review:**

Human reviewers examine AI suggestions and provide corrections.

**Review Team Tasks:**
- Review AI-suggested matches
- Correct errors
- Resolve ambiguous cases
- Review 40% of unmatched places independently

**Output:**
- `AI_review/cleaned_output.csv` - Corrected AI results
- `AI_review/user/*.csv` - Independent user reviews (40% sample)

### 6. Evaluate AI Performance

**Compare AI Results to Human Review:**

Compare human decision and AI-based predictions using Levenshtein distance.

**Script:** `AI-human_compare.py`

**Features:**
- Compares AI output vs user-reviewed results
- Calculates similarity at multiple thresholds (0.98, 0.95, 0.85)
- Evaluates state, county, and city separately


**Input:**
- `ai_outputs_all.csv` - AI predictions
- `user_reviewed.csv` - Human-corrected data

**Output:**
- `detailed_evaluation_results.csv` - All records with match scores
- `evaluation_statistics.csv` - Summary metrics
- `evaluation_report.txt` - Human-readable report

**Usage:**
```python
main(
    ai_output_file="data/ai_outputs_all.csv",
    user_reviewed_file="data/user_reviewed.csv",
    output_dir="evaluation_results"
)
```