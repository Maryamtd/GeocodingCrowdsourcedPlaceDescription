"""
Deterministic Exact-match toponym resolution.
Processes GEDCOM files, extracts birthplace information, and matches them
against reference data (states, counties, cities).
Results are saved to CSV files.
"""

import re
import os
import csv
import zipfile
import pandas as pd
from datetime import datetime
from collections import defaultdict
from ged4py.parser import GedcomReader

# ============================================================================
# STATE ABBREVIATIONS AND REFERENCE DATA
# ============================================================================

std_abbreviations = {
    'ak': 'alaska', 'al': 'alabama', 'ar': 'arkansas', 'az': 'arizona',
    'ca': 'california', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida',
    'il': 'illinois', 'ill': 'illinois', 'in': 'indiana', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'ma': 'massachusetts', 'md': 'maryland',
    'me': 'maine', 'mi': 'michigan', 'mo': 'missouri', 'ms': 'mississippi',
    'mt': 'montana', 'nc': 'north carolina', 'nd': 'north dakota',
    'ne': 'nebraska', 'nh': 'new hampshire', 'nj': 'new jersey',
    'nm': 'new mexico', 'nv': 'nevada', 'ny': 'new york', 'oh': 'ohio',
    'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island',
    'sc': 'south carolina', 'tn': 'tennessee', 'tx': 'texas', 'vt': 'vermont',
    'wi': 'wisconsin', 'wv': 'west virginia', 'va': 'virginia',
    'wa': 'washington', 'ia': 'iowa', 'ut': 'utah', 'wy': 'wyoming',
    'id': 'idaho', 'ga': 'georgia', 'dt': 'dakota', 'mn': 'minnesota',
    'dc': 'district of columbia', 'sd': 'south dakota', 'nt': 'northwest territory'
}

abbreviations = {
    'alaska': 'ak', 'alabama': 'al', 'arkansas': 'ar', 'arizona': 'az',
    'california': 'ca', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl',
    'illinois': 'il', 'indiana': 'in', 'kansas': 'ks', 'kentucky': 'ky',
    'louisiana': 'la', 'massachusetts': 'ma', 'maryland': 'md', 'maine': 'me',
    'michigan': 'mi', 'missouri': 'mo', 'mississippi': 'ms', 'montana': 'mt',
    'north carolina': 'nc', 'north dakota': 'nd', 'nebraska': 'ne',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm',
    'nevada': 'nv', 'new york': 'ny', 'ohio': 'oh', 'oklahoma': 'ok',
    'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri',
    'south carolina': 'sc', 'tennessee': 'tn', 'texas': 'tx', 'vermont': 'vt',
    'wisconsin': 'wi', 'west virginia': 'wv', 'virginia': 'va',
    'washington': 'wa', 'iowa': 'ia', 'utah': 'ut', 'wyoming': 'wy',
    'idaho': 'id', 'georgia': 'ga', 'dakota': 'dt', 'minnesota': 'mn',
    'district of columbia': 'dc', 'south dakota': 'sd',
    # Common typos and variations
    'wva': 'wv', 'ill': 'il', 'mass': 'ma', 'conn': 'ct', 'penn': 'pa',
    'penna': 'pa', 'tenn': 'tn', 'calif': 'ca', 'wisc': 'wi',
    'virgina': 'va', 'viginia': 'va', 'viriginia': 'va',
    'south carloina': 'sc', 'south caroina': 'sc',
    'northwest territory': 'nt'
}

territories = [
    "northwest territory", "southwest territory", "indiana territory",
    "mississippi territory", "louisiana territory", "missouri territory",
    "arkansas territory", "michigan territory", "florida territory",
    "wisconsin territory", "iowa territory", "oregon territory",
    "minnesota territory", "new mexico territory", "utah territory",
    "washington territory", "kansas territory", "nebraska territory",
    "colorado territory", "dakota territory", "arizona territory",
    "idaho territory", "montana territory", "wyoming territory",
    "oklahoma territory", "hawaii territory", "alaska territory"
]

state_reference_set = set(abbreviations.values()) | set(abbreviations.keys())
state_reference_set.update(territories)
states_reference_list = sorted(list(state_reference_set))

# Countries list for filtering foreign-born individuals
countries = [
    "afghanistan", "albania", "algeria", "andorra", "angola", "argentina",
    "armenia", "australia", "austria", "azerbaijan", "bahamas", "bahrain",
    "bangladesh", "barbados", "belarus", "belgium", "belize", "benin",
    "bolivia", "bosnia", "herzegovina", "botswana", "brazil", "britain",
    "great britain", "brunei", "bulgaria", "burundi", "cambodia", "cameroon",
    "canada", "chad", "chile", "china", "colombia", "costa rica", "croatia",
    "cuba", "cyprus", "denmark", "ecuador", "egypt", "england", "estonia",
    "ethiopia", "fiji", "finland", "france", "gabon", "gambia", "germany",
    "ghana", "greece", "guatemala", "guinea", "haiti", "honduras", "hungary",
    "iceland", "india", "indonesia", "iran", "iraq", "ireland", "israel",
    "italy", "jamaica", "japan", "jordan", "kenya", "korea", "kuwait",
    "latvia", "lebanon", "liberia", "libya", "lithuania", "luxembourg",
    "madagascar", "malaysia", "malta", "mexico", "morocco", "myanmar",
    "namibia", "nepal", "netherlands", "new zealand", "nicaragua", "niger",
    "nigeria", "norway", "oman", "pakistan", "palestine", "panama",
    "paraguay", "peru", "philippines", "poland", "portugal", "qatar",
    "romania", "russia", "rwanda", "saudi arabia", "scotland", "senegal",
    "serbia", "singapore", "slovakia", "slovenia", "somalia", "south africa",
    "spain", "sri lanka", "sudan", "sweden", "switzerland", "syria",
    "tanzania", "thailand", "tunisia", "turkey", "uganda", "ukraine",
    "united kingdom", "uruguay", "venezuela", "vietnam", "wales", "yemen",
    "zambia", "zimbabwe", "bavaria", "holland", "ontario", "quebec"
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def place_preprocess(place):
    """Standardize place names: lowercase, remove punctuation and numbers."""
    if not isinstance(place, str):
        return ''
    place = place.lower().strip()
    place = re.sub(r'[^\w\s]', ' ', place)
    place = re.sub(r'\d+', ' ', place)
    place = re.sub(' +', ' ', place).strip()
    return place


def not_foreign_born(place):
    """Check if birthplace contains foreign country names."""
    place_lower = place.lower()
    for country in countries:
        if country in place_lower:
            # Make sure it's not a state that happens to match
            if country not in state_reference_set:
                return False
    return True


def find_full_state_name(state_part):
    """Find full state name and abbreviation from input."""
    normalized_state = state_part.lower().strip()

    # Try direct match first
    direct_match = std_abbreviations.get(normalized_state)
    if direct_match:
        return [direct_match, normalized_state]

    # Search in abbreviations
    for full_name, abbr in abbreviations.items():
        if re.search(r'\b' + full_name + r'\b', normalized_state):
            std_abbr = abbr
            full_state_name = std_abbreviations.get(std_abbr, '')
            return [full_state_name, std_abbr]

    return [None, None]


def determine_census_year(birth_year):
    """Determine which census year to use based on birth year."""
    if birth_year < 1800:
        return 1800
    elif birth_year < 1810:
        return 1810
    elif birth_year < 1820:
        return 1820
    elif birth_year < 1830:
        return 1830
    elif birth_year < 1840:
        return 1840
    elif birth_year < 1850:
        return 1850
    elif birth_year < 1860:
        return 1860
    elif birth_year < 1870:
        return 1870
    elif birth_year < 1880:
        return 1880
    elif birth_year < 1900:
        return 1900
    elif birth_year < 1910:
        return 1910
    elif birth_year < 1920:
        return 1920
    elif birth_year < 1930:
        return 1930
    elif birth_year < 1940:
        return 1940
    else:
        return 1940


# ============================================================================
# PLACE MATCHING FUNCTION
# ============================================================================

def match_place(standard_birthplace, counties_df, cities_df, census_year):
    """
    Match birthplace against reference data and categorize the match.
    Returns: (category, matched_data_dict)
    """

    # Skip places ending with " co" (ambiguous: Colorado or County?)
    if standard_birthplace.endswith(" co"):
        return 'no_match', {
            'stdbirthplace': standard_birthplace,
            'census_year': census_year
        }

    # Find state in birthplace
    state_found = None
    last_match = None
    last_match_end = -1

    for item in reversed(states_reference_list):
        pattern = r'\b{}\b'.format(re.escape(item))
        matches = list(re.finditer(pattern, standard_birthplace, re.IGNORECASE))
        if matches and matches[-1].end() > last_match_end:
            last_match = matches[-1]
            last_match_end = last_match.end()
            state_found = last_match.group()

    if not state_found:
        return 'no_match', {
            'stdbirthplace': standard_birthplace,
            'census_year': census_year
        }

    # Parse state information
    start, end = last_match.span()
    state_level_birthplace = (standard_birthplace[:start] + ';' +
                              standard_birthplace[start:end] +
                              standard_birthplace[end:])

    name_abbr_list = find_full_state_name(state_found)
    state_found_full = name_abbr_list[0]

    if not state_found_full:
        return 'no_match', {
            'stdbirthplace': standard_birthplace,
            'census_year': census_year
        }

    # Check if there's content before the state
    pattern = r"^[^;]*\w"
    if not re.search(pattern, state_level_birthplace.lstrip()):
        # Only state found
        return 'state_only', {
            'birthplace': state_level_birthplace,
            'state': state_found,
            'stdstate': state_found_full,
            'census_year': census_year,
            'stdbirthplace': standard_birthplace
        }

    # Try to match county
    pre_semicolon_str = state_level_birthplace.split(';')[0]

    # Filter counties by state
    state_counties = counties_df[counties_df['state'] == state_found_full]

    county_found = None
    if not state_counties.empty:
        last_match_end = -1
        last_match = None

        for county in state_counties['county'].unique():
            pattern = r'\b{}\b'.format(re.escape(county))
            matches = list(re.finditer(pattern, pre_semicolon_str))
            if matches and matches[-1].end() > last_match_end:
                last_match = matches[-1]
                last_match_end = last_match.end()
                county_found = county

        if county_found:
            start = last_match.start()
            second_part = state_level_birthplace.split(";")[1] if len(
                state_level_birthplace.split(";")) > 1 else ""
            county_level_birthplace = (pre_semicolon_str[:start] + ";" +
                                       pre_semicolon_str[start:] + ";" +
                                       second_part).lstrip()

            # Try to match city
            pre_semicolon_str = county_level_birthplace.split(';')[0]
            if re.search(r"^[^;]*\w", pre_semicolon_str.lstrip()):
                state_county_cities = cities_df[
                    (cities_df['state'] == state_found_full) &
                    (cities_df['county'] == county_found)
                    ]

                city_found = None
                if not state_county_cities.empty:
                    for city in state_county_cities['stdtownship'].unique():
                        pattern = r'\b{}\b'.format(re.escape(city))
                        if re.search(pattern, pre_semicolon_str):
                            city_found = city
                            break

                if city_found:
                    return 'city_county_state', {
                        'birthplace': county_level_birthplace,
                        'city': city_found,
                        'county': county_found,
                        'state': state_found,
                        'stdstate': state_found_full,
                        'census_year': census_year,
                        'stdbirthplace': standard_birthplace
                    }
                else:
                    return 'county_state', {
                        'birthplace': county_level_birthplace,
                        'county': county_found,
                        'state': state_found,
                        'stdstate': state_found_full,
                        'census_year': census_year,
                        'stdbirthplace': standard_birthplace
                    }
            else:
                return 'county_state', {
                    'birthplace': county_level_birthplace,
                    'county': county_found,
                    'state': state_found,
                    'stdstate': state_found_full,
                    'census_year': census_year,
                    'stdbirthplace': standard_birthplace
                }

    # County not found, try to match city directly with state
    state_cities = cities_df[cities_df['state'] == state_found_full]

    city_found = None
    if not state_cities.empty:
        for city in state_cities['stdtownship'].unique():
            pattern = r'\b{}\b'.format(re.escape(city))
            if re.search(pattern, pre_semicolon_str):
                city_found = city
                break

    if city_found:
        return 'city_state', {
            'birthplace': state_level_birthplace,
            'city': city_found,
            'state': state_found,
            'stdstate': state_found_full,
            'census_year': census_year,
            'stdbirthplace': standard_birthplace
        }

    # Only state found
    return 'state_only', {
        'birthplace': state_level_birthplace,
        'state': state_found,
        'stdstate': state_found_full,
        'census_year': census_year,
        'stdbirthplace': standard_birthplace
    }


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================
def load_reference_data(census_year, counties_dir, cities_dir):
    """
    Load reference data for a specific census year.

    Args:
        census_year: Census year (e.g., 1850, 1860)
        counties_dir: Directory containing county reference CSV files
        cities_dir: Directory containing city reference CSV files

    Returns:
        tuple: (counties_df, cities_df)
    """
    counties_file = os.path.join(counties_dir, f'{census_year}.csv')
    cities_file = os.path.join(cities_dir, f'{census_year}.csv')

    counties_df = pd.DataFrame(columns=['state', 'county'])
    cities_df = pd.DataFrame(columns=['state', 'county', 'stdtownship'])

    # Load counties reference
    if os.path.exists(counties_file):
        counties_df = pd.read_csv(counties_file)
        # Preprocess place names
        counties_df['state'] = counties_df['state'].apply(place_preprocess)
        counties_df['county'] = counties_df['county'].apply(place_preprocess)
        print(f"  Loaded {len(counties_df)} counties for {census_year}")
    else:
        print(f"  Warning: Counties file not found for {census_year}: {counties_file}")

    # Load cities reference
    if os.path.exists(cities_file):
        cities_df = pd.read_csv(cities_file)
        # Preprocess place names
        cities_df['state'] = cities_df['state'].apply(place_preprocess)
        cities_df['county'] = cities_df['county'].apply(place_preprocess)
        cities_df['stdtownship'] = cities_df['stdtownship'].apply(place_preprocess)
        print(f"  Loaded {len(cities_df)} cities for {census_year}")
    else:
        print(f"  Warning: Cities file not found for {census_year}: {cities_file}")

    return counties_df, cities_df


def process_gedcom_files(zip_path, reference_dir, output_dir):
    """
    Process GEDCOM files from a zip archive.

    Args:
        zip_path: Path to zip file containing GEDCOM files
        reference_dir: Directory containing reference CSV files for counties/cities
        output_dir: Directory to save output CSV files
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load reference data
    print("Loading reference data...")

    # Cache for reference data by census year
    reference_cache = {}

    # Initialize result containers
    place_results = defaultdict(list)
    person_records = []

    # Statistics
    stats = {
        'total_individuals': 0,
        'no_birthplace': 0,
        'no_birthyear': 0,
        'foreign_born': 0,
        'matched': defaultdict(int)
    }

    print(f"Processing {zip_path}...")

    # Process the zip file
    try:
        with zipfile.ZipFile(zip_path) as zip_archive:
            for gedcom_name in zip_archive.namelist():
                if not gedcom_name.endswith(".ged"):
                    continue

                print(f"  Processing {gedcom_name}...")

                with zip_archive.open(gedcom_name) as gedcom_file:
                    with GedcomReader(gedcom_file, errors='ignore') as gedcom_reader:
                        for person in gedcom_reader.records0("INDI"):
                            stats['total_individuals'] += 1

                            birth_record = person.sub_tag("BIRT")
                            if birth_record:
                                birth_date = birth_record.sub_tag("DATE")
                                birthplace = birth_record.sub_tag("PLAC")
                                if birth_date and birth_date.value:
                                    if len(str(birth_date.value).split()) > 0:
                                        birth_year_str = birth_place_parts[-1]
                                        if "/" in birth_year_str:
                                            first_part = birth_year_str.split("/")[0]
                                            if first_part.isdigit():
                                                birth_year = int(first_part)
                                            else:
                                                birth_year = -1
                                        elif birth_year_str.isdigit():
                                            birth_year = int(birth_year_str)
                                        else:
                                            birth_year = -1
                                    else:
                                        birth_year = -1
                                else:
                                    birth_year = -1
                                continue

                            if not birth_record.sub_tag("PLAC"):
                                birthplace = ''
                                stats['no_birthplace'] += 1
                                continue
                            if birth_year == -1 or not (1000 < birth_year < 2030):
                                stats['no_birthyear'] += 1
                                continue

                            # Standardize birthplace
                            std_birthplace = place_preprocess(birthplace)

                            # Check if foreign-born
                            if not not_foreign_born(std_birthplace):
                                stats['foreign_born'] += 1
                                continue

                            # Determine census year
                            census_year = determine_census_year(birth_year)

                            # Load reference data for this census year (with caching)
                            if census_year not in reference_cache:
                                counties_df, cities_df = load_reference_data(
                                    census_year, counties_dir, cities_dir
                                )
                                reference_cache[census_year] = (counties_df, cities_df)
                            else:
                                counties_df, cities_df = reference_cache[census_year]

                            # Match place
                            category, match_data = match_place(
                                std_birthplace, counties_df, cities_df, census_year
                            )

                            stats['matched'][category] += 1
                            place_results[category].append(match_data)

                            if person:
                                indi_id = person.xref_id.replace("@", "").split("I")[-1]
                                gedcom_id = f"{os.path.splitext(gedcom_name)}_{indi_id}"
                                first_name = person.name.given if person.name.given else ""
                                last_name = person.name.surname if person.name.surname else ""
                                father_first_name = person.father.name.given if person.father.name.given else ""
                                father_last_name = person.father.name.surname if person.father.name.surname else ""
                                mother_first_name = person.mother.name.given if person.mother.name.given else ""
                                mother_last_name = person.mother.name.surname if person.mother.name.surname else ""
                                if person.sub_tag("SEX") and person.sub_tag("SEX").value:
                                    gender_value = person.sub_tag("SEX").value

                            # Store person record
                            person_record = {
                                'gedcom_id': gedcom_id,
                                'firstname': first_name,
                                'lastname': last_name,
                                'fathername': f"{father_first_name} {father_last_name}",
                                'mothername':  f"{mother_first_name} {mother_last_name}",
                                'gender': gender,
                                'birth_year': birth_year,
                                'stdbirthplace': std_birthplace,
                                'match_category': category
                            }
                            person_records.append(person_record)

    except Exception as e:
        print(f"Error processing zip file: {e}")
        return

    # Save results to CSV files
    print("\nSaving results...")

    # Save place matching results
    for category, records in place_results.items():
        if records:
            df = pd.DataFrame(records)
            output_file = os.path.join(output_dir, f'{category}.csv')
            df.to_csv(output_file, index=False)
            print(f"  Saved {len(records)} records to {category}.csv")

    # Save person records
    if person_records:
        persons_df = pd.DataFrame(person_records)
        persons_file = os.path.join(output_dir, 'persons.csv')
        persons_df.to_csv(persons_file, index=False)
        print(f"  Saved {len(person_records)} person records to persons.csv")

    # Save statistics
    stats_file = os.path.join(output_dir, 'processing_stats.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Processing Statistics\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Total Individuals: {stats['total_individuals']}\n")
        f.write(f"No Birthplace: {stats['no_birthplace']}\n")
        f.write(f"No Birth Year: {stats['no_birthyear']}\n")
        f.write(f"Foreign Born: {stats['foreign_born']}\n")
        f.write(f"\nPlace Matching Results:\n")
        for category, count in stats['matched'].items():
            f.write(f"  {category}: {count}\n")

    print(f"\nProcessing complete! Results saved to {output_dir}")
    print(f"\nStatistics:")
    print(f"  Total individuals: {stats['total_individuals']}")
    print(f"  Successfully matched: {sum(stats['matched'].values())}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    zip_path = "data/task/meuser30.zip"
    reference_dir = "data/references"
    output_dir = "data/output_results"

    print("GEDCOM Place Matching System")
    print("=" * 60)
    print("\nThis processes GEDCOM files and matches")
    print("birthplaces against reference data without.")
    print("\nResults are saved as CSV files.")
    print("=" * 60)

    # Uncomment to run:
    # process_gedcom_files(zip_path, reference_dir, output_dir)

    print("\nTo use this script:")
    print("1. Prepare your GEDCOM zip file")
    print("2. Organize reference CSV files by year:")
    print("   - counties_references/1850.csv (columns: state, county)")
    print("   - city_references/1850.csv (columns: state, county, stdtownship)")
    print("3. Update the paths above")
    print("4. Uncomment the function call")
    print("5. Run the script")