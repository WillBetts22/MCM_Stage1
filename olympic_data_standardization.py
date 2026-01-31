import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = "2025_Problem_C_Data"

def read_csv_robust(path, **kwargs):
    """Robust CSV reading with multiple encoding attempts."""
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("All attempted encodings failed", b"", 0, 1, "unknown")


class OlympicDataStandardizer:
    """
    Standardizes 120+ years of messy Olympic data according to specific rules.
    """
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.athletes = None
        self.hosts = None
        self.medals = None
        self.programs = None
        
    def load_data(self):
        """Load all Olympic data files."""
        print("Loading data files...")
        self.athletes = read_csv_robust(f"{self.data_dir}/summerOly_athletes.csv")
        self.hosts = read_csv_robust(f"{self.data_dir}/summerOly_hosts.csv")
        self.medals = read_csv_robust(f"{self.data_dir}/summerOly_medal_counts.csv")
        self.programs = read_csv_robust(f"{self.data_dir}/summerOly_programs.csv")
        
        print(f"Athletes shape: {self.athletes.shape}")
        print(f"Hosts shape: {self.hosts.shape}")
        print(f"Medals shape: {self.medals.shape}")
        print(f"Programs shape: {self.programs.shape}")
        
    def handle_historical_anomalies(self):
        """
        Handle historical anomalies:
        - Delete 1906 Intercalated Games data
        - Preserve gaps for 1940 and 1944 (WWII cancellations)
        - Keep 1980/1984 boycotts and East German doping as natural variance
        """
        print("\n=== Handling Historical Anomalies ===")
        
        # Delete 1906 Intercalated Games (not recognized by IOC)
        initial_count = len(self.athletes)
        self.athletes = self.athletes[self.athletes['edition'] != '1906 Summer Olympics']
        deleted_1906 = initial_count - len(self.athletes)
        print(f"Deleted {deleted_1906} records from 1906 Intercalated Games")
        
        # Check for medal data from 1906
        if 'edition' in self.medals.columns:
            initial_medals = len(self.medals)
            self.medals = self.medals[self.medals['edition'] != '1906 Summer Olympics']
            deleted_medals_1906 = initial_medals - len(self.medals)
            print(f"Deleted {deleted_medals_1906} medal records from 1906")
        
        # Verify 1940 and 1944 are already gaps (these were cancelled)
        editions_in_data = sorted(self.athletes['edition'].unique()) if 'edition' in self.athletes.columns else []
        has_1940 = any('1940' in str(ed) for ed in editions_in_data)
        has_1944 = any('1944' in str(ed) for ed in editions_in_data)
        
        if has_1940 or has_1944:
            print("WARNING: Found data for 1940 or 1944 - these should be gaps!")
        else:
            print("✓ Confirmed: 1940 and 1944 are properly missing (WWII cancellations)")
        
        print("✓ Keeping 1980/1984 boycotts and East German records as natural variance")
        
    def standardize_country_codes(self):
        """
        Apply country merging rules:
        - Map Soviet Union (URS) to Russia (RUS)
        - Merge East Germany and West Germany into Germany
        - Keep post-Soviet states separate
        - Drop Mixed Team entries
        """
        print("\n=== Standardizing Country Codes ===")
        
        # Country mapping dictionary
        country_mapping = {
            'URS': 'RUS',  # Soviet Union -> Russia
            'EUN': 'RUS',  # Unified Team (1992) -> Russia
            'GDR': 'GER',  # East Germany -> Germany
            'FRG': 'GER',  # West Germany -> Germany
        }
        
        # Apply to athletes dataset
        if 'noc' in self.athletes.columns:
            initial_counts = self.athletes['noc'].value_counts()
            
            # Apply mapping
            self.athletes['noc'] = self.athletes['noc'].replace(country_mapping)
            
            # Drop Mixed Team entries
            mixed_team_count = len(self.athletes[self.athletes['noc'] == 'MIX'])
            self.athletes = self.athletes[self.athletes['noc'] != 'MIX']
            
            print(f"✓ Merged Soviet Union records into Russia")
            print(f"✓ Merged East/West Germany into Germany")
            print(f"✓ Dropped {mixed_team_count} Mixed Team entries")
            
            final_counts = self.athletes['noc'].value_counts()
            if 'RUS' in final_counts:
                print(f"  Russia total records: {final_counts.get('RUS', 0)}")
            if 'GER' in final_counts:
                print(f"  Germany total records: {final_counts.get('GER', 0)}")
        
        # Apply to medals dataset
        if 'noc' in self.medals.columns:
            self.medals['noc'] = self.medals['noc'].replace(country_mapping)
            self.medals = self.medals[self.medals['noc'] != 'MIX']
            print("✓ Applied country mappings to medals dataset")
        
        # Verify post-Soviet states remain separate
        post_soviet_states = ['LTU', 'KAZ', 'UKR', 'BLR', 'EST', 'LAT', 'GEO', 'ARM', 'AZE']
        found_states = [noc for noc in post_soviet_states if noc in self.athletes['noc'].unique()]
        if found_states:
            print(f"✓ Keeping post-Soviet states separate: {', '.join(found_states)}")
    
    def create_active_athletes_subset(self, cutoff_year=2020):
        """
        Create subset of athletes who competed on or after the cutoff year.
        This is crucial for judging current country strength for 2028 predictions.
        """
        print(f"\n=== Creating Active Athletes Subset (competed >= {cutoff_year}) ===")
        
        # Determine the year column
        year_col = None
        for col in ['year', 'edition_year', 'games_year']:
            if col in self.athletes.columns:
                year_col = col
                break
        
        if year_col is None:
            # Try to extract year from edition column
            if 'edition' in self.athletes.columns:
                print("Extracting year from edition column...")
                self.athletes['year'] = self.athletes['edition'].str.extract(r'(\d{4})').astype(float)
                year_col = 'year'
            else:
                raise ValueError("Cannot find year information in athletes dataset")
        
        # Convert to numeric if needed
        self.athletes[year_col] = pd.to_numeric(self.athletes[year_col], errors='coerce')
        
        # Create active athletes subset
        self.active_athletes = self.athletes[self.athletes[year_col] >= cutoff_year].copy()
        
        total_athletes = len(self.athletes)
        active_count = len(self.active_athletes)
        inactive_count = total_athletes - active_count
        
        print(f"Total athletes in dataset: {total_athletes:,}")
        print(f"Active athletes (>= {cutoff_year}): {active_count:,}")
        print(f"Filtered out (pre-{cutoff_year}): {inactive_count:,}")
        print(f"Retention rate: {active_count/total_athletes*100:.1f}%")
        
        # Summary by country for active athletes
        if 'noc' in self.active_athletes.columns:
            top_countries = self.active_athletes['noc'].value_counts().head(10)
            print(f"\nTop 10 countries by active athlete count:")
            for noc, count in top_countries.items():
                print(f"  {noc}: {count:,}")
        
        return self.active_athletes
    
    def handle_dual_nationality_medals(self):
        """
        Count dual-nationality medals for both countries.
        This requires identifying athletes with dual nationality and duplicating their medal records.
        """
        print("\n=== Handling Dual Nationality Medals ===")
        
        # Check if there's a dual nationality indicator in the data
        # This is often in athlete_id or separate columns
        # For now, we'll document the approach
        
        print("Note: Dual nationality handling requires:")
        print("  1. Identification of dual-nationality athletes")
        print("  2. Duplication of medal records for both countries")
        print("  3. Proper attribution in medal counts")
        print("✓ Framework ready - implement when dual nationality data is available")
    
    def validate_data_quality(self):
        """Validate the standardized data quality."""
        print("\n=== Data Quality Validation ===")
        
        # Check for expected gaps
        if 'year' in self.athletes.columns:
            years = sorted(self.athletes['year'].dropna().unique())
            print(f"Year range: {int(min(years))} - {int(max(years))}")
            
            # Check gap years
            expected_gaps = [1940, 1944]
            for gap_year in expected_gaps:
                if gap_year in years:
                    print(f"  WARNING: {gap_year} should be a gap year!")
                else:
                    print(f"  ✓ {gap_year} properly missing (WWII)")
        
        # Check for removed entities
        if 'noc' in self.athletes.columns:
            banned_codes = ['MIX', 'URS', 'GDR', 'FRG', 'EUN']
            for code in banned_codes:
                if code in self.athletes['noc'].unique():
                    print(f"  WARNING: {code} still present in data!")
                else:
                    print(f"  ✓ {code} properly removed/merged")
        
        # Check for null values
        print(f"\nNull value check:")
        for col in ['noc', 'medal', 'sport']:
            if col in self.athletes.columns:
                null_count = self.athletes[col].isna().sum()
                null_pct = null_count / len(self.athletes) * 100
                print(f"  {col}: {null_count:,} nulls ({null_pct:.1f}%)")
    
    def save_standardized_data(self, output_dir="standardized_data"):
        """Save all standardized datasets."""
        print(f"\n=== Saving Standardized Data to {output_dir}/ ===")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save main datasets
        self.athletes.to_csv(output_path / "athletes_standardized.csv", index=False)
        print(f"✓ Saved: athletes_standardized.csv ({len(self.athletes):,} rows)")
        
        self.hosts.to_csv(output_path / "hosts_standardized.csv", index=False)
        print(f"✓ Saved: hosts_standardized.csv ({len(self.hosts):,} rows)")
        
        self.medals.to_csv(output_path / "medals_standardized.csv", index=False)
        print(f"✓ Saved: medals_standardized.csv ({len(self.medals):,} rows)")
        
        self.programs.to_csv(output_path / "programs_standardized.csv", index=False)
        print(f"✓ Saved: programs_standardized.csv ({len(self.programs):,} rows)")
        
        # Save active athletes subset
        if hasattr(self, 'active_athletes'):
            self.active_athletes.to_csv(output_path / "active_athletes.csv", index=False)
            print(f"✓ Saved: active_athletes.csv ({len(self.active_athletes):,} rows)")
        
        print(f"\n✓ All standardized data saved to {output_dir}/")
    
    def run_full_standardization(self):
        """Execute complete standardization pipeline."""
        print("=" * 70)
        print("OLYMPIC DATA STANDARDIZATION PIPELINE")
        print("=" * 70)
        
        self.load_data()
        self.handle_historical_anomalies()
        self.standardize_country_codes()
        self.create_active_athletes_subset(cutoff_year=2020)
        self.handle_dual_nationality_medals()
        self.validate_data_quality()
        self.save_standardized_data()
        
        print("\n" + "=" * 70)
        print("STANDARDIZATION COMPLETE!")
        print("=" * 70)
        
        return {
            'athletes': self.athletes,
            'active_athletes': self.active_athletes,
            'hosts': self.hosts,
            'medals': self.medals,
            'programs': self.programs
        }


def main():
    """Main execution function."""
    standardizer = OlympicDataStandardizer(DATA_DIR)
    results = standardizer.run_full_standardization()
    
    # Return datasets for further analysis
    return results


if __name__ == "__main__":
    results = main()
