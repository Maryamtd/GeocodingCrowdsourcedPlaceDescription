"""
Levenshtein Distance Evaluation for Place Matching
Compares AI output to user input for state, county, and city using CSV files.
Calculates match accuracy at different similarity thresholds.
"""

import pandas as pd
import logging
import os
import re
import unicodedata
from Levenshtein import ratio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def standardize_values(value):
    """
    Standardize values for comparison by normalizing text, 
    removing punctuation, and handling abbreviations.
    
    Args:
        value: The place name value to standardize
        
    Returns:
        Standardized string or None if invalid
    """
    # Check for invalid/null values
    if not value or str(value).lower() in [
        'nan', 'null', 'n/a', 'no', '', 'none', 'unknown', 
        'probably', 'unclear', 'ambiguous', 'prob', 'likely'
    ]:
        return None

    # Convert to lowercase and strip whitespace
    value = str(value).strip().lower()

    # Normalize unicode characters (remove accents)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('utf-8')

    # Replace common abbreviations with standard forms
    replacements = {
        r'\bsaint\b': 'st',
        r'\bmount\b': 'mt',
        r'\bfort\b': 'ft',
        r'\bdoctor\b': 'dr',
        r'\bavenue\b': 'ave',
        r'\bboulevard\b': 'blvd',
        r'\broad\b': 'rd',
    }
    
    for pattern, replacement in replacements.items():
        value = re.sub(pattern, replacement, value)

    # Remove 'city' and 'county' suffixes
    value = re.sub(r'\b(city|county|cty|co)\b', '', value)

    # Replace directional words with abbreviations
    directional_replacements = {
        r'\bnorth\b': 'n',
        r'\bsouth\b': 's',
        r'\beast\b': 'e',
        r'\bwest\b': 'w',
        r'\bnortheast\b': 'ne',
        r'\bsoutheast\b': 'se',
        r'\bnorthwest\b': 'nw',
        r'\bsouthwest\b': 'sw',
    }
    
    for pattern, replacement in directional_replacements.items():
        value = re.sub(pattern, replacement, value)

    # Remove all punctuation
    value = re.sub(r'[^\w\s]', '', value)

    # Remove extra spaces
    value = re.sub(r'\s+', ' ', value).strip()

    return value


def calculate_levenshtein_ratio(val1, val2):
    """
    Calculate the Levenshtein ratio between two values.
    
    Args:
        val1: First value to compare
        val2: Second value to compare
        
    Returns:
        Float between 0 and 1 representing similarity (1 = identical)
    """
    if not val1 or not val2:
        return 0.0
    return ratio(val1, val2)


def evaluate_distance(row, thresholds):
    """
    Evaluate distance for state, county, and city at different thresholds.
    
    Args:
        row: DataFrame row containing AI and user values
        thresholds: List of similarity thresholds to test
        
    Returns:
        Dictionary of match results for each place type and threshold
    """
    results = {}
    
    for place_type in ['state', 'county', 'city']:
        ai_value = row[f'ai_{place_type}']
        user_value = row[f'user_{place_type}']

        # Calculate Levenshtein ratio
        levenshtein_ratio = calculate_levenshtein_ratio(ai_value, user_value)
        
        # Store the ratio itself for analysis
        results[f'{place_type}_ratio'] = levenshtein_ratio

        # Check match at each threshold
        for threshold in thresholds:
            match_key = f'{place_type}_match_{threshold}'
            results[match_key] = 1 if levenshtein_ratio >= threshold else 0

    return results


def load_data_from_csv(ai_output_file, user_reviewed_file):
    """
    Load AI output and user-reviewed data from CSV files.
    
    Args:
        ai_output_file: Path to CSV with AI predictions
        user_reviewed_file: Path to CSV with user corrections
        
    Returns:
        Merged DataFrame with both AI and user data
    """
    logger.info(f"Loading AI output from: {ai_output_file}")
    ai_df = pd.read_csv(ai_output_file)
    
    logger.info(f"Loading user reviewed data from: {user_reviewed_file}")
    user_df = pd.read_csv(user_reviewed_file)
    
    # Rename columns to distinguish AI vs user values
    ai_df = ai_df.rename(columns={
        'state': 'ai_state',
        'county': 'ai_county',
        'city': 'ai_city'
    })
    
    user_df = user_df.rename(columns={
        'state': 'user_state',
        'county': 'user_county',
        'city': 'user_city'
    })
    
    # Merge on standardized birthplace
    logger.info("Merging AI and user data on stdbirthplace...")
    df = pd.merge(
        ai_df[['stdbirthplace', 'ai_state', 'ai_county', 'ai_city']],
        user_df[['stdbirthplace', 'user_state', 'user_county', 'user_city']],
        on='stdbirthplace',
        how='inner'
    )
    
    logger.info(f"Successfully merged {len(df)} records")
    
    return df


def calculate_statistics(df, thresholds):
    """
    Calculate match statistics at different thresholds.
    
    Args:
        df: DataFrame with match results
        thresholds: List of thresholds used
        
    Returns:
        Dictionary of statistics
    """
    total_records = len(df)
    stats = {}

    # Calculate statistics for each place type and threshold
    for threshold in thresholds:
        for place_type in ['state', 'county', 'city']:
            match_key = f'{place_type}_match_{threshold}'
            num_matches = df[match_key].sum()
            ratio_matches = num_matches / total_records if total_records > 0 else 0
            
            stats[f'{place_type}_matches_{threshold}'] = {
                'num_matches': int(num_matches),
                'ratio_matches': ratio_matches
            }

    # Calculate whole record matches (all three fields match)
    for threshold in thresholds:
        total_match_key = f'total_matches_{threshold}'
        df[total_match_key] = df[[
            f'state_match_{threshold}',
            f'county_match_{threshold}',
            f'city_match_{threshold}'
        ]].sum(axis=1)
        
        whole_matches = (df[total_match_key] == 3).sum()
        ratio_whole_matches = whole_matches / total_records if total_records > 0 else 0
        
        stats[f'whole_record_matches_{threshold}'] = {
            'whole_matches': int(whole_matches),
            'ratio_whole_matches': ratio_whole_matches
        }

    return stats, df


def save_results(df, stats, output_dir):
    """
    Save evaluation results to CSV files.
    
    Args:
        df: DataFrame with all evaluation results
        stats: Dictionary of calculated statistics
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    detailed_output = os.path.join(output_dir, 'detailed_evaluation_results.csv')
    df.to_csv(detailed_output, index=False)
    logger.info(f"Saved detailed results to: {detailed_output}")
    
    # Save summary statistics
    stats_output = os.path.join(output_dir, 'evaluation_statistics.csv')
    
    stats_rows = []
    for key, value in stats.items():
        if isinstance(value, dict):
            row = {'metric': key}
            row.update(value)
            stats_rows.append(row)
    
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(stats_output, index=False)
    logger.info(f"Saved statistics to: {stats_output}")
    
    # Save summary report
    report_output = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_output, 'w') as f:
        f.write("PLACE MATCHING EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Records Evaluated: {len(df)}\n")
        f.write(f"Evaluation Date: {pd.Timestamp.now()}\n\n")
        
        # Extract thresholds from stats keys
        thresholds = sorted(set([
            float(key.split('_')[-1]) 
            for key in stats.keys() 
            if 'match' in key and key.split('_')[-1].replace('.', '').isdigit()
        ]))
        
        for threshold in thresholds:
            f.write(f"\nRESULTS AT THRESHOLD {threshold}\n")
            f.write("-" * 70 + "\n")
            
            for place_type in ['state', 'county', 'city']:
                key = f'{place_type}_matches_{threshold}'
                if key in stats:
                    num = stats[key]['num_matches']
                    pct = stats[key]['ratio_matches'] * 100
                    f.write(f"{place_type.capitalize():12} matches: {num:6d} ({pct:6.2f}%)\n")
            
            whole_key = f'whole_record_matches_{threshold}'
            if whole_key in stats:
                num = stats[whole_key]['whole_matches']
                pct = stats[whole_key]['ratio_whole_matches'] * 100
                f.write(f"{'Whole record':12} matches: {num:6d} ({pct:6.2f}%)\n")
    
    logger.info(f"Saved evaluation report to: {report_output}")


def main(ai_output_file, user_reviewed_file, output_dir='evaluation_results'):
    """
    Main function to run the evaluation.
    
    Args:
        ai_output_file: Path to CSV file with AI predictions
                       (columns: stdbirthplace, state, county, city)
        user_reviewed_file: Path to CSV file with user corrections
                           (columns: stdbirthplace, state, county, city)
        output_dir: Directory to save results
    """
    try:
        # Load data from CSV files
        df = load_data_from_csv(ai_output_file, user_reviewed_file)
        
        # Standardize all place name columns
        logger.info("Standardizing place names...")
        for col in ['ai_state', 'ai_county', 'ai_city', 'user_state', 'user_county', 'user_city']:
            df[col] = df[col].apply(standardize_values)
        
        # Set thresholds for evaluation
        thresholds = [0.98, 0.95, 0.85]
        logger.info(f"Evaluating at thresholds: {thresholds}")
        
        # Evaluate Levenshtein distance for each row
        logger.info("Calculating Levenshtein distances...")
        df_results = df.apply(
            lambda row: pd.Series(evaluate_distance(row, thresholds)), 
            axis=1
        )
        df = pd.concat([df, df_results], axis=1)
        
        # Calculate statistics
        logger.info("Calculating statistics...")
        stats, df = calculate_statistics(df, thresholds)
        
        # Save results
        save_results(df, stats, output_dir)
        
        # Log summary to console
        total_records = len(df)
        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUATION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total Records: {total_records}")
        
        for threshold in thresholds:
            logger.info(f"\nThreshold {threshold}:")
            for place_type in ['state', 'county', 'city']:
                key = f'{place_type}_matches_{threshold}'
                num = stats[key]['num_matches']
                pct = stats[key]['ratio_matches'] * 100
                logger.info(f"  {place_type.capitalize():10} matches: {num:6d} ({pct:6.2f}%)")
            
            whole_key = f'whole_record_matches_{threshold}'
            num = stats[whole_key]['whole_matches']
            pct = stats[whole_key]['ratio_whole_matches'] * 100
            logger.info(f"  {'Whole record':10} matches: {num:6d} ({pct:6.2f}%)")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Results saved to: {output_dir}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    ai_output_file = "data/ai_outputs_all.csv"
    user_reviewed_file = "data/user_reviewed.csv"
    output_dir = "evaluation_results"
    
    print("\nPlace Matching Evaluation Tool (CSV Version)")
    print("=" * 70)
    print("\nThis tool evaluates AI place matching accuracy against user reviews.")
    print("\nRequired CSV files:")
    print("  1. AI Output CSV (columns: stdbirthplace, state, county, city)")
    print("  2. User Reviewed CSV (columns: stdbirthplace, state, county, city)")
    print("\nThe tool calculates Levenshtein similarity ratios and reports")
    print("match accuracy at multiple thresholds (0.98, 0.95, 0.85)")
    print("=" * 70)
    
    # Uncomment to run:
    # main(ai_output_file, user_reviewed_file, output_dir)
    
    print("\nTo use this script:")
    print("1. Prepare AI output CSV with columns: stdbirthplace, state, county, city")
    print("2. Prepare user reviewed CSV with same columns")
    print("3. Update file paths above")
    print("4. Uncomment the main() call")
    print("5. Run the script")
    print("\nOutput files:")
    print("  - detailed_evaluation_results.csv (all records with match scores)")
    print("  - evaluation_statistics.csv (summary statistics)")
    print("  - evaluation_report.txt (human-readable report)")