import pandas as pd
import numpy as np
from pathlib import Path


def analyze_country_strength(active_athletes_df, medals_df=None):
    """
    Analyze current country strength based on active athletes.
    Used for predicting 2028 Olympic performance.
    """
    print("=" * 70)
    print("COUNTRY STRENGTH ANALYSIS (For 2028 Predictions)")
    print("=" * 70)
    
    # Medal count by country (active athletes only)
    if 'medal' in active_athletes_df.columns and 'noc' in active_athletes_df.columns:
        # Filter for actual medals (exclude NA)
        medalists = active_athletes_df[active_athletes_df['medal'].notna()].copy()
        
        # Count by medal type
        medal_counts = medalists.groupby(['noc', 'medal']).size().unstack(fill_value=0)
        
        # Calculate total medals and weighted score
        if 'Gold' in medal_counts.columns:
            medal_counts['Total'] = medal_counts.sum(axis=1)
            # Weighted score: Gold=3, Silver=2, Bronze=1
            medal_counts['Weighted_Score'] = (
                medal_counts.get('Gold', 0) * 3 + 
                medal_counts.get('Silver', 0) * 2 + 
                medal_counts.get('Bronze', 0) * 1
            )
            
            # Sort by weighted score
            medal_counts = medal_counts.sort_values('Weighted_Score', ascending=False)
            
            print("\nTop 20 Countries by Medal Performance (Active Athletes >= 2020):")
            print(medal_counts.head(20))
        else:
            print("\nMedal column format not recognized")
    
    # Athlete count by country
    athlete_counts = active_athletes_df.groupby('noc').size().sort_values(ascending=False)
    print(f"\nTop 20 Countries by Active Athlete Count:")
    print(athlete_counts.head(20))
    
    # Sport diversity (indicator of program strength)
    if 'sport' in active_athletes_df.columns:
        sport_diversity = active_athletes_df.groupby('noc')['sport'].nunique().sort_values(ascending=False)
        print(f"\nTop 20 Countries by Sport Diversity:")
        print(sport_diversity.head(20))
    
    return {
        'medal_counts': medal_counts if 'medal' in active_athletes_df.columns else None,
        'athlete_counts': athlete_counts,
        'sport_diversity': sport_diversity if 'sport' in active_athletes_df.columns else None
    }


def verify_historical_cleaning(athletes_df):
    """
    Verify that historical anomalies were properly handled.
    """
    print("\n" + "=" * 70)
    print("HISTORICAL CLEANING VERIFICATION")
    print("=" * 70)
    
    # Check for 1906
    if 'edition' in athletes_df.columns:
        has_1906 = athletes_df['edition'].str.contains('1906', na=False).any()
        print(f"1906 Intercalated Games present: {'❌ YES (ERROR!)' if has_1906 else '✓ NO (CORRECT)'}")
    
    # Check for war years
    if 'year' in athletes_df.columns:
        years = athletes_df['year'].unique()
        has_1940 = 1940 in years
        has_1944 = 1944 in years
        print(f"1940 data present: {'❌ YES (ERROR!)' if has_1940 else '✓ NO (CORRECT - WWII gap)'}")
        print(f"1944 data present: {'❌ YES (ERROR!)' if has_1944 else '✓ NO (CORRECT - WWII gap)'}")
    
    # Check for merged/removed country codes
    if 'noc' in athletes_df.columns:
        nocs = athletes_df['noc'].unique()
        print("\nCountry Code Verification:")
        
        removed_codes = {
            'URS': 'Soviet Union (should be merged to RUS)',
            'GDR': 'East Germany (should be merged to GER)',
            'FRG': 'West Germany (should be merged to GER)',
            'EUN': 'Unified Team (should be merged to RUS)',
            'MIX': 'Mixed Team (should be removed)'
        }
        
        for code, desc in removed_codes.items():
            present = code in nocs
            status = '❌ ERROR' if present else '✓ CORRECT'
            print(f"  {code}: {status} - {desc}")
        
        # Check that merged codes exist
        merged_codes = {'RUS': 'Russia', 'GER': 'Germany'}
        for code, desc in merged_codes.items():
            present = code in nocs
            status = '✓ PRESENT' if present else '❌ MISSING'
            print(f"  {code}: {status} - {desc}")


def get_data_summary(athletes_df, active_athletes_df=None):
    """
    Generate comprehensive summary of the standardized data.
    """
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal Athletes: {len(athletes_df):,}")
    
    if 'year' in athletes_df.columns:
        year_range = f"{int(athletes_df['year'].min())} - {int(athletes_df['year'].max())}"
        print(f"Year Range: {year_range}")
        print(f"Number of Olympic editions: {athletes_df['year'].nunique()}")
    
    if 'noc' in athletes_df.columns:
        print(f"Number of countries: {athletes_df['noc'].nunique()}")
    
    if 'sport' in athletes_df.columns:
        print(f"Number of sports: {athletes_df['sport'].nunique()}")
    
    if 'event' in athletes_df.columns:
        print(f"Number of events: {athletes_df['event'].nunique()}")
    
    if active_athletes_df is not None:
        print(f"\nActive Athletes (>= 2020): {len(active_athletes_df):,}")
        retention_pct = len(active_athletes_df) / len(athletes_df) * 100
        print(f"Retention rate: {retention_pct:.1f}%")


def export_for_modeling(standardized_dir="standardized_data", output_dir="modeling_data"):
    """
    Export cleaned data in formats optimized for modeling.
    """
    print("\n" + "=" * 70)
    print("EXPORTING DATA FOR MODELING")
    print("=" * 70)
    
    input_path = Path(standardized_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load standardized data
    athletes = pd.read_csv(input_path / "athletes_standardized.csv")
    active_athletes = pd.read_csv(input_path / "active_athletes.csv")
    
    # Create modeling datasets
    
    # 1. Country-level features for 2028 prediction
    print("\n1. Creating country-level features...")
    country_features = create_country_features(active_athletes)
    country_features.to_csv(output_path / "country_features_2028.csv", index=True)
    print(f"   ✓ Saved: country_features_2028.csv ({len(country_features)} countries)")
    
    # 2. Historical medal trends
    print("\n2. Creating historical medal trends...")
    medal_trends = create_medal_trends(athletes)
    medal_trends.to_csv(output_path / "medal_trends_historical.csv", index=False)
    print(f"   ✓ Saved: medal_trends_historical.csv ({len(medal_trends)} records)")
    
    # 3. Sport-specific country strength
    print("\n3. Creating sport-specific strength indicators...")
    sport_strength = create_sport_strength(active_athletes)
    sport_strength.to_csv(output_path / "sport_strength_by_country.csv", index=False)
    print(f"   ✓ Saved: sport_strength_by_country.csv ({len(sport_strength)} records)")
    
    print(f"\n✓ All modeling datasets saved to {output_dir}/")


def create_country_features(active_df):
    """Create country-level features from active athletes."""
    features = []
    
    for noc in active_df['noc'].unique():
        country_data = active_df[active_df['noc'] == noc]
        
        feat = {
            'noc': noc,
            'athlete_count': len(country_data),
            'unique_sports': country_data['sport'].nunique() if 'sport' in country_data else 0,
            'unique_events': country_data['event'].nunique() if 'event' in country_data else 0,
        }
        
        # Medal counts
        if 'medal' in country_data.columns:
            medals = country_data[country_data['medal'].notna()]
            feat['gold_count'] = len(medals[medals['medal'] == 'Gold'])
            feat['silver_count'] = len(medals[medals['medal'] == 'Silver'])
            feat['bronze_count'] = len(medals[medals['medal'] == 'Bronze'])
            feat['total_medals'] = len(medals)
        
        features.append(feat)
    
    return pd.DataFrame(features).set_index('noc')


def create_medal_trends(athletes_df):
    """Create historical medal trends by country and year."""
    if 'medal' in athletes_df.columns and 'year' in athletes_df.columns:
        medalists = athletes_df[athletes_df['medal'].notna()]
        trends = medalists.groupby(['year', 'noc', 'medal']).size().reset_index(name='count')
        return trends
    return pd.DataFrame()


def create_sport_strength(active_df):
    """Create sport-specific strength indicators."""
    if 'sport' in active_df.columns and 'noc' in active_df.columns:
        # Count medals by sport and country
        sport_strength = active_df.groupby(['noc', 'sport']).agg({
            'medal': lambda x: x.notna().sum(),  # Medal count
            'athlete_id': 'count' if 'athlete_id' in active_df else 'size'  # Athlete count
        }).reset_index()
        sport_strength.columns = ['noc', 'sport', 'medals', 'athletes']
        return sport_strength
    return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    print("Utility functions loaded. Import this module to use the analysis functions.")
    print("\nAvailable functions:")
    print("  - analyze_country_strength()")
    print("  - verify_historical_cleaning()")
    print("  - get_data_summary()")
    print("  - export_for_modeling()")
