# Phase 1 Complete Consolidated Script
# Kaggle Meta Analysis - All-in-One Foundation Builder
# Incorporates: Investigation + Loading + Fixing + Validation + Research Prep

import os
import sys
import pandas as pd
import numpy as np
import psutil
import gc
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class KaggleMetaFoundationBuilder:
    """Complete Phase 1 foundation builder - all-in-one solution"""
    
    def __init__(self, base_path=None):
        # Auto-detect current directory if no path provided
        if base_path is None:
            self.base_path = Path.cwd()
        else:
            self.base_path = Path(base_path)
        
        self.archive_path = self.base_path / "archive"
        
        # Validate that archive directory exists
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Archive directory not found: {self.archive_path}")
        
        # Create phase directories
        self.phase_dirs = {}
        for phase in range(1, 5):
            phase_dir = self.base_path / f"phase_{phase}"
            try:
                phase_dir.mkdir(parents=True, exist_ok=True)
                self.phase_dirs[f"phase_{phase}"] = phase_dir
                
                # Create subdirectories for each phase
                for subdir in ["data", "analysis", "plots", "results"]:
                    (phase_dir / subdir).mkdir(parents=True, exist_ok=True)
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not create {phase_dir}: {e}")
        
        # Current phase
        self.current_phase = 1
        self.current_dir = self.phase_dirs.get(f"phase_{self.current_phase}")
        
        # Data storage
        self.datasets = {}
        self.load_statistics = {}
        self.file_mappings = {}
        self.investigation_results = {}
        
        # System info
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.safe_memory_threshold = 0.75
        
        print(f"🏗️  Kaggle Meta Foundation Builder Initialized")
        print(f"📁 Base Path: {self.base_path}")
        print(f"📊 Archive Path: {self.archive_path}")
        print(f"💾 System Memory: {self.system_memory_gb:.1f} GB")
        print(f"🔄 Current Phase: {self.current_phase}")
        
        # Verify archive files exist
        csv_files = list(self.archive_path.glob("*.csv"))
        print(f"📋 CSV files found: {len(csv_files)}")

    def investigate_file_structure(self, filename, max_rows=3):
        """Investigate actual column names and data types in a CSV file"""
        
        filepath = self.archive_path / filename
        if not filepath.exists():
            return {"error": f"File not found: {filename}"}
        
        file_size_gb = filepath.stat().st_size / (1024**3)
        
        try:
            # Read just the header and a few rows
            sample_df = pd.read_csv(filepath, nrows=max_rows)
            
            # Get column information
            column_info = []
            for col in sample_df.columns:
                col_info = {
                    "name": col,
                    "dtype": str(sample_df[col].dtype),
                    "sample_values": sample_df[col].head(3).tolist(),
                    "has_nulls": sample_df[col].isnull().any(),
                    "is_likely_date": any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated'])
                }
                column_info.append(col_info)
            
            # Identify potential date columns
            date_columns = [col["name"] for col in column_info if col["is_likely_date"]]
            
            result = {
                "filename": filename,
                "file_size_gb": file_size_gb,
                "total_columns": len(sample_df.columns),
                "column_names": list(sample_df.columns),
                "column_info": column_info,
                "potential_date_columns": date_columns,
                "sample_data_shape": sample_df.shape
            }
            
            return result
            
        except Exception as e:
            return {"error": str(e), "filename": filename}

    def categorize_all_files(self):
        """Categorize all CSV files and investigate problematic ones"""
        
        print(f"\n📊 COMPREHENSIVE FILE CATEGORIZATION & INVESTIGATION")
        print(f"=" * 70)
        
        # File categories based on our analysis
        file_categories = {
            "core_entities": {
                "description": "Main entity tables - Load fully",
                "files": ["Users.csv", "Competitions.csv", "Teams.csv", "Organizations.csv", "Tags.csv"]
            },
            "large_transaction_tables": {
                "description": "High-volume transaction data - Intelligent sampling required",
                "files": ["Submissions.csv", "KernelVersions.csv", "UserAchievements.csv", "ForumMessages.csv"]
            },
            "relationship_networks": {
                "description": "Junction tables - Medium sampling",
                "files": ["KernelVersionCompetitionSources.csv", "KernelVersionDatasetSources.csv", 
                         "TeamMemberships.csv", "UserFollowers.csv", "KernelTags.csv", "DatasetTags.csv"]
            },
            "content_repositories": {
                "description": "Content tables - Strategic sampling",
                "files": ["Kernels.csv", "Datasets.csv", "DatasetVersions.csv", "Models.csv"]
            }
        }
        
        # Investigate key files to understand their structure
        key_files_to_investigate = [
            "UserAchievements.csv", "KernelVersions.csv", "Submissions.csv", 
            "Users.csv", "Competitions.csv", "ForumMessages.csv"
        ]
        
        print(f"🔍 Investigating key files for column structure:")
        for filename in key_files_to_investigate:
            print(f"   🔍 Investigating {filename}...")
            result = self.investigate_file_structure(filename)
            self.investigation_results[filename] = result
            
            if "error" not in result:
                date_cols = result.get("potential_date_columns", [])
                print(f"      📅 Date columns: {date_cols}")
                print(f"      📊 {result['total_columns']} columns, {result['file_size_gb']:.3f} GB")
            else:
                print(f"      ❌ Error: {result['error']}")
        
        # Create file mappings with correct date columns
        self.file_mappings = {}
        for filename, result in self.investigation_results.items():
            if "error" not in result:
                self.file_mappings[filename] = result.get("potential_date_columns", [])
        
        return file_categories

    def get_optimal_sample_size(self, file_size_gb, base_sample_size=1000000):
        """Calculate optimal sample size based on file size and available memory"""
        
        # Memory-based calculation
        available_memory_gb = self.system_memory_gb * self.safe_memory_threshold
        memory_per_million_records = 0.5  # GB
        max_records_by_memory = int((available_memory_gb / memory_per_million_records) * 1000000)
        
        # File size-based calculation
        if file_size_gb < 0.1:  # < 100MB
            return None  # Load fully
        elif file_size_gb < 1.0:  # < 1GB
            return min(base_sample_size * 2, max_records_by_memory)
        elif file_size_gb < 5.0:  # < 5GB
            return min(base_sample_size * 5, max_records_by_memory)
        else:  # >= 5GB
            return min(base_sample_size * 10, max_records_by_memory)

    def optimize_dataframe_memory(self, df):
        """Optimize DataFrame memory usage"""
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (initial_memory - final_memory) / initial_memory * 100
        
        return df, reduction

    def load_dataset_smart(self, filename, sample_size=None, parse_dates=None):
        """Smart dataset loading with memory optimization and correct column handling"""
        
        filepath = self.archive_path / filename
        
        if not filepath.exists():
            print(f"❌ File not found: {filename}")
            return None
        
        file_size_gb = filepath.stat().st_size / (1024**3)
        
        # Use investigated date columns if available
        if filename in self.file_mappings and parse_dates is None:
            parse_dates = self.file_mappings[filename]
        
        print(f"📁 Loading {filename} ({file_size_gb:.3f} GB)...")
        if parse_dates:
            print(f"   📅 Date columns: {parse_dates}")
        
        # Determine sample size if not provided
        if sample_size is None:
            sample_size = self.get_optimal_sample_size(file_size_gb)
        
        # Memory check before loading
        memory_before = psutil.virtual_memory().percent
        
        try:
            start_time = datetime.now()
            
            if sample_size is None:
                # Full load
                df = pd.read_csv(filepath, parse_dates=parse_dates)
                load_type = "FULL"
            else:
                # Sampled load
                df = pd.read_csv(filepath, nrows=sample_size, parse_dates=parse_dates)
                load_type = f"SAMPLE ({sample_size:,})"
            
            # Optimize memory usage
            df, memory_reduction = self.optimize_dataframe_memory(df)
            
            load_time = (datetime.now() - start_time).total_seconds()
            memory_after = psutil.virtual_memory().percent
            
            # Store statistics
            self.load_statistics[filename] = {
                "load_type": load_type,
                "records_loaded": len(df),
                "file_size_gb": file_size_gb,
                "load_time_seconds": load_time,
                "memory_reduction_percent": memory_reduction,
                "memory_before_percent": memory_before,
                "memory_after_percent": memory_after,
                "columns": len(df.columns),
                "date_columns_used": parse_dates if parse_dates else [],
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"   ✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
            print(f"   🚀 Load time: {load_time:.1f}s ({len(df)/load_time:,.0f} rows/sec)")
            print(f"   💾 Memory optimization: {memory_reduction:.1f}% reduction")
            print(f"   📊 Memory usage: {memory_before:.1f}% → {memory_after:.1f}%")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {str(e)}")
            return None

    def load_all_core_datasets(self):
        """Load all core datasets with intelligent handling"""
        
        print(f"\n🚀 LOADING ALL CORE DATASETS")
        print(f"=" * 50)
        
        # Core entities - load fully
        core_entities = {
            "Users.csv": "users",
            "Competitions.csv": "competitions", 
            "Teams.csv": "teams",
            "Organizations.csv": "organizations",
            "Tags.csv": "tags"
        }
        
        print(f"📊 Loading core entities:")
        for filename, dataset_name in core_entities.items():
            df = self.load_dataset_smart(filename)
            if df is not None:
                self.datasets[dataset_name] = df
        
        # Large transaction tables - intelligent sampling
        large_tables = {
            "Submissions.csv": ("submissions", 3000000),    # 3M sample
            "UserAchievements.csv": ("userachievements", 5000000),  # 5M sample
            "KernelVersions.csv": ("kernelversions", 3000000),      # 3M sample
            "ForumMessages.csv": ("forummessages", 2000000)         # 2M sample
        }
        
        print(f"\n📊 Loading large transaction tables:")
        for filename, (dataset_name, sample_size) in large_tables.items():
            df = self.load_dataset_smart(filename, sample_size=sample_size)
            if df is not None:
                self.datasets[dataset_name] = df
        
        # Trigger garbage collection
        gc.collect()
        
        print(f"\n✅ Core dataset loading complete!")
        print(f"📊 Datasets loaded: {len(self.datasets)}")
        print(f"💾 Current memory usage: {psutil.virtual_memory().percent:.1f}%")

    def validate_all_datasets(self):
        """Comprehensive validation of all loaded datasets"""
        
        print(f"\n🔍 COMPREHENSIVE DATASET VALIDATION")
        print(f"=" * 60)
        
        validation_results = {}
        total_records = 0
        total_memory_mb = 0
        
        for name, df in self.datasets.items():
            if df is None:
                validation_results[name] = {"status": "failed", "error": "Dataset is None"}
                continue
                
            print(f"\n📋 Validating {name}...")
            
            validation = {
                "status": "success",
                "total_records": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
                "null_counts": df.isnull().sum().to_dict(),
                "duplicate_rows": df.duplicated().sum(),
                "date_columns": [],
                "numeric_columns": [],
                "categorical_columns": []
            }
            
            # Analyze column types
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]':
                    validation["date_columns"].append(col)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    validation["numeric_columns"].append(col)
                elif df[col].dtype == 'object':
                    validation["categorical_columns"].append(col)
            
            validation_results[name] = validation
            total_records += validation['total_records']
            total_memory_mb += validation['memory_usage_mb']
            
            print(f"   📊 Records: {validation['total_records']:,}")
            print(f"   📈 Columns: {validation['total_columns']} (📅 {len(validation['date_columns'])} dates, 🔢 {len(validation['numeric_columns'])} numeric)")
            print(f"   💾 Memory: {validation['memory_usage_mb']:.1f} MB")
            print(f"   🔍 Duplicates: {validation['duplicate_rows']:,}")
            
            if validation['date_columns']:
                print(f"   📅 Date columns: {validation['date_columns']}")
            
            # Show columns with high null rates
            high_null_cols = {k: v for k, v in validation['null_counts'].items() if v > len(df) * 0.1}
            if high_null_cols:
                print(f"   ⚠️  High null columns (>10%): {len(high_null_cols)}")
        
        validation_summary = {
            "total_datasets": len(self.datasets),
            "successful_datasets": len([v for v in validation_results.values() if v.get("status") == "success"]),
            "total_records": total_records,
            "total_memory_mb": total_memory_mb,
            "system_memory_percent": psutil.virtual_memory().percent
        }
        
        print(f"\n🏆 VALIDATION SUMMARY:")
        print(f"   📊 Datasets loaded: {validation_summary['successful_datasets']}/{validation_summary['total_datasets']}")
        print(f"   📈 Total records: {validation_summary['total_records']:,}")
        print(f"   💾 Total memory: {validation_summary['total_memory_mb']:.1f} MB")
        print(f"   🖥️  System memory: {validation_summary['system_memory_percent']:.1f}%")
        
        return validation_results, validation_summary

    def create_research_ready_datasets(self):
        """Prepare datasets specifically for research analysis"""
        
        print(f"\n🔬 CREATING RESEARCH-READY DATASETS")
        print(f"=" * 50)
        
        research_datasets = {}
        
        # 1. Competition Evolution Analysis
        if 'competitions' in self.datasets and 'submissions' in self.datasets:
            competitions = self.datasets['competitions'].copy()
            submissions = self.datasets['submissions'].copy()
            
            # Clean and prepare for research
            competitions['EnabledYear'] = competitions['EnabledDate'].dt.year
            competitions_clean = competitions[
                (competitions['EnabledYear'] >= 2009) & 
                (competitions['EnabledYear'] <= 2025)
            ].copy()
            
            submissions['SubmissionYear'] = submissions['SubmissionDate'].dt.year
            submissions_clean = submissions[
                (submissions['SubmissionYear'] >= 2009) & 
                (submissions['SubmissionYear'] <= 2025)
            ].copy()
            
            research_datasets['competition_evolution'] = {
                'competitions': competitions_clean,
                'submissions': submissions_clean,
                'description': 'Anti-leakage evolution and competition sophistication analysis',
                'research_questions': [
                    'How did submission limits evolve over time?',
                    'What is the timeline of evaluation sophistication?',
                    'Can we prove Kaggle developed anti-leakage before academic recognition?'
                ]
            }
            
            print(f"   ✅ Competition Evolution: {len(competitions_clean):,} competitions, {len(submissions_clean):,} submissions")
        
        # 2. User Journey & Elite Development
        if 'users' in self.datasets and 'userachievements' in self.datasets:
            users = self.datasets['users'].copy()
            achievements = self.datasets['userachievements'].copy()
            
            # Clean and prepare for research
            users['RegisterYear'] = users['RegisterDate'].dt.year
            users_clean = users[users['RegisterYear'] >= 2009].copy()
            
            achievements['AchievementYear'] = achievements['TierAchievementDate'].dt.year
            achievements_clean = achievements[achievements['AchievementYear'] >= 2009].copy()
            
            research_datasets['user_journey'] = {
                'users': users_clean,
                'achievements': achievements_clean,
                'description': 'User progression, elite emergence, and community evolution',
                'research_questions': [
                    'How do users progress from novice to expert?',
                    'What patterns distinguish elite performers?',
                    'How has the community evolved geographically?'
                ]
            }
            
            print(f"   ✅ User Journey: {len(users_clean):,} users, {len(achievements_clean):,} achievements")
        
        # 3. Innovation Diffusion & Knowledge Transfer
        if 'kernelversions' in self.datasets and 'tags' in self.datasets:
            kernels = self.datasets['kernelversions'].copy()
            tags = self.datasets['tags'].copy()
            
            # Clean and prepare for research
            kernels['CreationYear'] = kernels['CreationDate'].dt.year
            kernels_clean = kernels[kernels['CreationYear'] >= 2009].copy()
            
            research_datasets['innovation_diffusion'] = {
                'kernelversions': kernels_clean,
                'tags': tags,
                'description': 'Innovation adoption patterns and knowledge transfer mechanisms',
                'research_questions': [
                    'How quickly do new techniques spread through the platform?',
                    'What are the most viral innovation patterns?',
                    'How does knowledge transfer from research to practice?'
                ]
            }
            
            print(f"   ✅ Innovation Diffusion: {len(kernels_clean):,} kernel versions, {len(tags):,} tags")
        
        return research_datasets

    def save_complete_foundation(self):
        """Save all results and create complete foundation"""
        
        print(f"\n💾 SAVING COMPLETE PHASE 1 FOUNDATION")
        print(f"=" * 60)
        
        try:
            # Save investigation results
            investigation_path = self.current_dir / "analysis" / "file_investigation_complete.json"
            with open(investigation_path, 'w') as f:
                json.dump(self.investigation_results, f, indent=2, default=str)
            print(f"🔍 Investigation results saved: {investigation_path}")
            
            # Save all datasets
            datasets_path = self.current_dir / "data" / "phase1_datasets_complete.pkl"
            with open(datasets_path, 'wb') as f:
                pickle.dump(self.datasets, f)
            print(f"💾 Complete datasets saved: {datasets_path}")
            
            # Save load statistics
            stats_path = self.current_dir / "results" / "load_statistics_complete.json"
            with open(stats_path, 'w') as f:
                json.dump(self.load_statistics, f, indent=2)
            print(f"📊 Load statistics saved: {stats_path}")
            
            # Create and save research datasets
            research_datasets = self.create_research_ready_datasets()
            research_path = self.current_dir / "data" / "research_ready_datasets.pkl"
            with open(research_path, 'wb') as f:
                pickle.dump(research_datasets, f)
            print(f"🔬 Research datasets saved: {research_path}")
            
            # Create final comprehensive summary
            final_summary = {
                "phase": "1_complete_consolidated",
                "completion_time": datetime.now().isoformat(),
                "system_info": {
                    "total_memory_gb": self.system_memory_gb,
                    "memory_usage_percent": psutil.virtual_memory().percent
                },
                "datasets_loaded": len(self.datasets),
                "total_records": sum(len(df) for df in self.datasets.values()),
                "total_memory_mb": sum(df.memory_usage(deep=True).sum() / 1024**2 for df in self.datasets.values()),
                "datasets_ready": list(self.datasets.keys()),
                "research_components": list(research_datasets.keys()) if research_datasets else [],
                "file_investigations": len(self.investigation_results),
                "foundation_complete": len(self.datasets) >= 8,  # Expect at least 8 core datasets
                "next_phase": "Phase 2: Competition Evolution & Anti-Leakage Analysis",
                "research_thesis": "AI Competitions as Gold Standard for GenAI Evaluation",
                "evidence_ready": {
                    "competition_evolution": 'competition_evolution' in research_datasets,
                    "user_journey": 'user_journey' in research_datasets,
                    "innovation_diffusion": 'innovation_diffusion' in research_datasets
                }
            }
            
            # Save final summary
            summary_path = self.current_dir / "results" / "phase1_foundation_complete.json"
            with open(summary_path, 'w') as f:
                json.dump(final_summary, f, indent=2)
            print(f"📋 Final summary saved: {summary_path}")
            
            return final_summary
            
        except Exception as e:
            print(f"❌ Error saving foundation: {e}")
            return None

def run_complete_phase1_foundation():
    """Execute the complete Phase 1 foundation building process"""
    
    print("🏗️  KAGGLE META ANALYSIS - COMPLETE PHASE 1 FOUNDATION")
    print("=" * 80)
    print("📋 Complete Phase 1 Goals:")
    print("   • Investigate all CSV file structures and column names")
    print("   • Load core datasets with correct column handling")
    print("   • Fix problematic datasets (UserAchievements, KernelVersions)")
    print("   • Validate all data quality and relationships")
    print("   • Create research-ready datasets for thesis validation")
    print("   • Establish complete foundation for Phase 2 analysis")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Initialize foundation builder
        builder = KaggleMetaFoundationBuilder()
        
        # Step 1: Categorize and investigate all files
        file_categories = builder.categorize_all_files()
        
        # Step 2: Load all core datasets
        builder.load_all_core_datasets()
        
        # Step 3: Validate everything
        validation_results, validation_summary = builder.validate_all_datasets()
        
        # Step 4: Save complete foundation
        final_summary = builder.save_complete_foundation()
        
        # Calculate total execution time
        total_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n🎉 COMPLETE PHASE 1 FOUNDATION ESTABLISHED!")
        print(f"=" * 80)
        
        if final_summary:
            print(f"🏆 FOUNDATION SUCCESS METRICS:")
            print(f"   📊 Datasets loaded: {final_summary['datasets_loaded']}")
            print(f"   📈 Total records: {final_summary['total_records']:,}")
            print(f"   💾 Memory usage: {final_summary['total_memory_mb']:.1f} MB ({final_summary['system_info']['memory_usage_percent']:.1f}% system)")
            print(f"   🔬 Research components: {len(final_summary['research_components'])}")
            print(f"   ⏱️  Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            print(f"   🎯 Foundation complete: {final_summary['foundation_complete']}")
            
            print(f"\n🔬 RESEARCH READINESS:")
            for component, ready in final_summary['evidence_ready'].items():
                status = "✅" if ready else "❌"
                print(f"   {status} {component.replace('_', ' ').title()}")
            
            print(f"\n🚀 READY FOR: {final_summary['next_phase']}")
            print(f"📄 THESIS: {final_summary['research_thesis']}")
        else:
            print(f"⚠️  Foundation completed with some issues - check error messages above")
        
        print(f"\n📁 All results saved in: {builder.current_dir}")
        print(f"🔄 Ready to proceed to Phase 2: Competition Evolution Analysis")
        
        return {
            'builder': builder,
            'validation_results': validation_results,
            'validation_summary': validation_summary,
            'final_summary': final_summary,
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"\n❌ PHASE 1 FOUNDATION FAILED: {str(e)}")
        print(f"🔧 Debug info:")
        print(f"   • Current directory: {Path.cwd()}")
        print(f"   • Archive exists: {Path('archive').exists()}")
        print(f"   • Files in archive: {len(list(Path('archive').glob('*'))) if Path('archive').exists() else 0}")
        raise

if __name__ == "__main__":
    results = run_complete_phase1_foundation()
