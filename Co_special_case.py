"""
Colorado "co co" Place Matching Handler
Handles birthplaces ending with "co co" by matching them against Colorado references.
Searches across multiple census years when needed.
Results are saved to CSV files.
"""

import re
import os
import pandas as pd
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def place_preprocess(place):
    """
    Standardize place names by removing punctuation, numbers, and extra spaces.

    Args:
        place: Raw place name string

    Returns:
        Standardized place name in lowercase
    """
    if not isinstance(place, str):
        return ''

    place = place.lower().strip()
    place = re.sub(r'[^\w\s]', ' ', place)  # Remove punctuation
    place = re.sub(r'\d+', ' ', place)  # Remove numbers
    place = re.sub(r'\s+', ' ', place).strip()  # Remove extra spaces
    return place


def load_reference_data(census_year, counties_dir, cities_dir):
    """
    Load reference data for a specific census year.

    Args:
        census_year: Census year (e.g., 1850, 1860)
        counties_dir: Directory containing county reference CSV files
        cities_dir: Directory containing city reference CSV files

    Returns:
        tuple: (counties_df, cities_df) with multi-level indexes
    """
    counties_file = os.path.join(counties_dir, f'{census_year}.csv')
    cities_file = os.path.join(cities_dir, f'{census_year}.csv')

    # Load counties
    if os.path.exists(counties_file):
        df_counties = pd.read_csv(counties_file)
        df_counties['state'] = df_counties['state'].apply(place_preprocess)
        df_counties['county'] = df_counties['county'].apply(place_preprocess)
        df_counties_indexed = df_counties.set_index(['state', 'county'], drop=False)
        df_counties_indexed = df_counties_indexed.sort_index()
        logger.debug(f"Loaded {len(df_counties)} counties for {census_year}")
    else:
        logger.warning(f"Counties file not found: {counties_file}")
        df_counties_indexed = pd.DataFrame(columns=['state', 'county']).set_index(['state', 'county'])

    # Load cities
    if os.path.exists(cities_file):
        df_cities = pd.read_csv(cities_file)
        df_cities['state'] = df_cities['state'].apply(place_preprocess)
        df_cities['county'] = df_cities['county'].apply(place_preprocess)
        df_cities['stdtownship'] = df_cities['stdtownship'].apply(place_preprocess)
        df_indexed_cities = df_cities.set_index(['state', 'county', 'stdtownship'], drop=False)
        df_indexed_cities = df_indexed_cities.sort_index()
        logger.debug(f"Loaded {len(df_cities)} cities for {census_year}")
    else:
        logger.warning(f"Cities file not found: {cities_file}")
        df_indexed_cities = pd.DataFrame(columns=['state', 'county', 'stdtownship']).set_index(
            ['state', 'county', 'stdtownship'])

    return df_counties_indexed, df_indexed_cities


def determine_next_year(initial_year, iteration):
    """
    Determine the next census year to search based on iteration.
    This allows searching in progressively later census years if no match is found.

    Args:
        initial_year: Original census year from birth year
        iteration: Current iteration number (0-13)

    Returns:
        Next census year to search, or None if no more years available
    """
    if iteration == 0:
        return initial_year
    elif iteration == 1:
        if initial_year == 1880:
            return 1900
        elif initial_year < 1940:
            return initial_year + 10
        else:
            return None
    elif iteration == 2:
        if initial_year in [1870, 1880]:
            return initial_year + 30
        elif initial_year < 1930:
            return initial_year + 20
        else:
            return None
    elif iteration == 3:
        if initial_year in [1860, 1870, 1880]:
            return initial_year + 40
        elif initial_year < 1920:
            return initial_year + 30
        else:
            return None
    elif iteration == 4:
        if initial_year in [1850, 1860, 1870, 1880]:
            return initial_year + 50
        elif initial_year < 1910:
            return initial_year + 40
        else:
            return None
    elif iteration == 5:
        if initial_year in [1840, 1850, 1860, 1870, 1880]:
            return initial_year + 60
        elif initial_year < 1840:
            return initial_year + 50
        else:
            return None
    elif iteration == 6:
        if initial_year < 1830:
            return initial_year + 60
        elif 1870 >= initial_year >= 1830:
            return initial_year + 70
        else:
            return None
    elif iteration == 7:
        if initial_year < 1820:
            return initial_year + 70
        elif 1860 >= initial_year >= 1820:
            return initial_year + 80
        else:
            return None
    elif iteration == 8:
        if initial_year < 1810:
            return initial_year + 80
        elif 1850 >= initial_year >= 1810:
            return initial_year + 90
        else:
            return None
    elif iteration == 9:
        if initial_year == 1790:
            return initial_year + 90
        elif 1840 >= initial_year >= 1800:
            return initial_year + 100
        else:
            return None
    elif iteration == 10:
        if initial_year < 1840:
            return initial_year + 110
        else:
            return None
    elif iteration == 11:
        if initial_year < 1830:
            return initial_year + 120
        else:
            return None
    elif iteration == 12:
        if initial_year < 1820:
            return initial_year + 130
        else:
            return None
    elif iteration == 13:
        if initial_year < 1810:
            return initial_year + 140
        else:
            return None

    return None


def match_colorado_place(stdbirthplace, census_year, reference_counties, reference_cities,
                         is_second_search=False, original_year=None):
    """
    Match a Colorado birthplace against reference data.

    Args:
        stdbirthplace: Standardized birthplace string ending with "co co"
        census_year: Census year for reference data
        reference_counties: County reference DataFrame
        reference_cities: City reference DataFrame
        is_second_search: Whether this is a secondary search in different year
        original_year: Original census year (for second search tracking)

    Returns:
        tuple: (category, match_data_dict)
    """
    state = 'co'
    state_found_full = 'colorado'

    # Remove "co co" and add delimiter
    state_level_birthplace = stdbirthplace[:-5].strip() + ';co'
    pre_semicolon_str = state_level_birthplace.split(';')[0]

    # Search for county
    county_found = None
    if state_found_full in reference_counties.index.get_level_values('state').unique():
        filtered_counties_df = reference_counties.loc[state_found_full]

        last_match_end = -1
        last_match = None

        for county in reversed(filtered_counties_df.index.get_level_values('county').unique()):
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

            # Search for city
            city_found = None
            pre_city_str = county_level_birthplace.split(';')[0]

            if (pre_city_str and
                    county_found in reference_cities.index.get_level_values('county').unique() and
                    (state_found_full, county_found) in reference_cities.index):

                filtered_cities_df = reference_cities.loc[(state_found_full, county_found)]

                for city in filtered_cities_df.index.get_level_values('stdtownship').unique():
                    pattern = r'\b{}\b'.format(re.escape(city))
                    if re.search(pattern, pre_city_str):
                        city_found = city
                        break

                if city_found:
                    result = {
                        'birthplace': county_level_birthplace,
                        'city': city_found,
                        'county': county_found,
                        'state': state,
                        'stdstate': state_found_full,
                        'stdbirthplace': stdbirthplace
                    }
                    if is_second_search:
                        result['census_year_byear'] = original_year
                        result['census_year'] = census_year
                        return 'second_search_city_county_state', result
                    else:
                        result['census_year'] = census_year
                        return 'city_county_state', result

            # County found, city not found
            result = {
                'birthplace': county_level_birthplace,
                'county': county_found,
                'state': state,
                'stdstate': state_found_full,
                'stdbirthplace': stdbirthplace
            }
            if is_second_search:
                result['census_year_byear'] = original_year
                result['census_year'] = census_year
                return 'second_search_county_and_state_nocity', result
            else:
                result['census_year'] = census_year
                return 'county_state', result

    # County not found, search for city directly
    if state_found_full in reference_cities.index.get_level_values('state').unique():
        filtered_cities_df = reference_cities.loc[state_found_full]

        city_found = None
        for city in reversed(filtered_cities_df.index.get_level_values('stdtownship').unique()):
            pattern = r'\b{}\b'.format(re.escape(city))
            if re.search(pattern, pre_semicolon_str):
                city_found = city
                break

        if city_found:
            result = {
                'birthplace': state_level_birthplace,
                'city': city_found,
                'state': state,
                'stdstate': state_found_full,
                'stdbirthplace': stdbirthplace
            }
            if is_second_search:
                result['census_year_byear'] = original_year
                result['census_year'] = census_year
                return 'second_search_city_and_state_nocounty', result
            else:
                result['census_year'] = census_year
                return 'city_and_state_nocounty', result

    # Only state found
    result = {
        'birthplace': state_level_birthplace,
        'state': state,
        'stdstate': state_found_full,
        'stdbirthplace': stdbirthplace,
        'census_year': census_year
    }
    if is_second_search:
        return 'no_match', result
    else:
        return 'state_nocounty_nocity', result


def process_colorado_birthplaces(input_file, counties_dir, cities_dir, output_dir,
                                 max_iterations=14):
    """
    Process birthplaces ending with "co co" (Colorado).
    Searches across multiple census years if no match found initially.

    Args:
        input_file: CSV file with unmatched birthplaces
                   (columns: stdbirthplace, census_year, count)
        counties_dir: Directory containing county reference CSVs by year
        cities_dir: Directory containing city reference CSVs by year
        output_dir: Directory to save results
        max_iterations: Maximum number of census years to search
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load input data
    logger.info(f"Loading input data from {input_file}")
    df_input = pd.read_csv(input_file)

    # Filter for birthplaces ending with "co co"
    df_colorado = df_input[df_input['stdbirthplace'].str.endswith(' co co')].copy()
    logger.info(f"Found {len(df_colorado)} birthplaces ending with 'co co'")

    if len(df_colorado) == 0:
        logger.warning("No birthplaces ending with 'co co' found")
        return

    # Cache for reference data
    reference_cache = {}

    # Results containers
    results = defaultdict(list)
    processed_records = set()

    # Statistics
    stats = {
        'only_state_count': 0,
        'city_co_st_level_count': 0,
        'county_state_level_count': 0,
        'city_state_count': 0
    }

    # Process each iteration
    for iteration in range(max_iterations):
        logger.info(f"Starting iteration {iteration}")

        records_to_process = []
        for _, row in df_colorado.iterrows():
            record_key = (row['stdbirthplace'], row['census_year'])
            if record_key not in processed_records:
                records_to_process.append(row)

        if not records_to_process:
            logger.info("All records processed")
            break

        for row in records_to_process:
            stdbirthplace = row['stdbirthplace']
            original_year = row['census_year']
            count = row.get('count', 1)

            # Determine which census year to search
            search_year = determine_next_year(original_year, iteration)

            if search_year is None:
                logger.debug(f"No more census years available for {original_year} at iteration {iteration}")
                continue

            # Load reference data for this year (with caching)
            if search_year not in reference_cache:
                counties_df, cities_df = load_reference_data(search_year, counties_dir, cities_dir)
                reference_cache[search_year] = (counties_df, cities_df)
            else:
                counties_df, cities_df = reference_cache[search_year]

            # Match the place
            is_second_search = (iteration > 0)
            category, match_data = match_colorado_place(
                stdbirthplace, search_year, counties_df, cities_df,
                is_second_search=is_second_search,
                original_year=original_year if is_second_search else None
            )

            # Add count to match data
            match_data['count'] = count

            # Store result if a match was found
            if category != 'no_match':
                results[category].append(match_data)
                processed_records.add((stdbirthplace, original_year))

                # Update statistics
                if 'city_county_state' in category:
                    stats['city_co_st_level_count'] += 1
                elif 'county' in category and 'state' in category:
                    stats['county_state_level_count'] += 1
                elif 'city' in category and 'state' in category:
                    stats['city_state_count'] += 1
                elif 'state' in category:
                    stats['only_state_count'] += 1

                logger.info(f"Matched {stdbirthplace} to {category} in year {search_year}")

    # Save results to CSV files
    logger.info("Saving results to CSV files...")

    for category, records in results.items():
        if records:
            df_result = pd.DataFrame(records)
            output_file = os.path.join(output_dir, f'{category}.csv')

            # Append to existing file or create new
            if os.path.exists(output_file):
                df_existing = pd.read_csv(output_file)
                df_result = pd.concat([df_existing, df_result], ignore_index=True)

            df_result.to_csv(output_file, index=False)
            logger.info(f"Saved {len(records)} records to {category}.csv")

    # Save statistics
    stats_file = os.path.join(output_dir, 'colorado_processing_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Colorado 'co co' Processing Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total records processed: {len(df_colorado)}\n")
        f.write(f"Successfully matched: {len(processed_records)}\n")
        f.write(f"Unmatched: {len(df_colorado) - len(processed_records)}\n\n")
        f.write(f"Match breakdown:\n")
        f.write(f"  City + County + State: {stats['city_co_st_level_count']}\n")
        f.write(f"  County + State: {stats['county_state_level_count']}\n")
        f.write(f"  City + State: {stats['city_state_count']}\n")
        f.write(f"  State only: {stats['only_state_count']}\n")

    logger.info(f"Statistics saved to {stats_file}")

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("COLORADO PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(df_colorado)}")
    logger.info(f"Successfully matched: {len(processed_records)}")
    logger.info(f"City + County + State matches: {stats['city_co_st_level_count']}")
    logger.info(f"County + State matches: {stats['county_state_level_count']}")
    logger.info(f"City + State matches: {stats['city_state_count']}")
    logger.info(f"State only matches: {stats['only_state_count']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Example usage
    input_file = "data/nostate_nocounty_nocity.csv"
    counties_dir = "counties_references"
    cities_dir = "city_references"
    output_dir = "colorado_results"

    print("\nColorado 'co co' Place Matching Handler")
    print("=" * 70)
    print("\nThis tool handles birthplaces ending with 'co co' (Colorado).")
    print("It searches across multiple census years to find the best match.")
    print("\nInput: CSV file with columns: stdbirthplace, census_year, count")
    print("       (filtered to records ending with ' co co')")
    print("\nReference data: Year-based CSV files in:")
    print("  - counties_references/1850.csv, 1860.csv, etc.")
    print("  - city_references/1850.csv, 1860.csv, etc.")
    print("=" * 70)

    # Uncomment to run:
    # process_colorado_birthplaces(input_file, counties_dir, cities_dir, output_dir)

    print("\nTo use this script:")
    print("1. Prepare input CSV with unmatched birthplaces")
    print("2. Filter or ensure it includes birthplaces ending with ' co co'")
    print("3. Organize reference CSVs by year")
    print("4. Update file paths above")
    print("5. Uncomment the function call")
    print("6. Run the script")
    print("\nOutput:")
    print("  - Multiple CSV files by match category")
    print("  - colorado_processing_stats.txt (summary statistics)")