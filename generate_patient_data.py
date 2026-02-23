#!/usr/bin/env python3
"""
Generate realistic patient-level data for rare disease analysis
Creates enriched clinical cohorts (not population-based) with sufficient cases for statistical analysis
"""

import pandas as pd
import numpy as np
import json
from synthetic_personal_data import generate_synthetic_personal_data

# Set seed for reproducibility
np.random.seed(42)

# Target number of cases per disease per node (enriched clinical cohorts)
# These represent specialized clinical centers, not population samples
CASES_PER_DISEASE = {
    "Cystic Fibrosis": 15,  # Enriched cohort from CF clinic
    "Huntington Disease": 10,  # HD specialty center
    "Duchenne Muscular Dystrophy": 12,  # Neuromuscular clinic
    "Sickle Cell Disease": 18,  # Sickle cell center
    "Hereditary Breast Cancer": 20,  # Cancer genetics clinic
    "Lynch Syndrome": 15,  # Hereditary cancer registry
}

# Pathogenic variants and their associations with diseases
# Format: variant_id: {gene, disease, odds_ratio, carrier_freq_in_cases, carrier_freq_in_controls}
PATHOGENIC_VARIANTS = {
    # Cystic Fibrosis variants (CFTR gene)
    "rs75961395": {"gene": "CFTR", "disease_name": "Cystic Fibrosis", 
                   "freq_in_cases": 0.40, "freq_in_controls": 0.002},
    "rs113993960": {"gene": "CFTR", "disease_name": "Cystic Fibrosis", 
                    "freq_in_cases": 0.65, "freq_in_controls": 0.003},
    "rs121908745": {"gene": "CFTR", "disease_name": "Cystic Fibrosis", 
                    "freq_in_cases": 0.35, "freq_in_controls": 0.001},
    "rs121909001": {"gene": "CFTR", "disease_name": "Cystic Fibrosis", 
                    "freq_in_cases": 0.25, "freq_in_controls": 0.001},
    
    # Huntington Disease variants (HTT gene)
    "rs121909298": {"gene": "HTT", "disease_name": "Huntington Disease", 
                    "freq_in_cases": 0.90, "freq_in_controls": 0.0001},
    "rs121909299": {"gene": "HTT", "disease_name": "Huntington Disease", 
                    "freq_in_cases": 0.10, "freq_in_controls": 0.00005},
    
    # Duchenne Muscular Dystrophy (DMD gene)
    "rs128625267": {"gene": "DMD", "disease_name": "Duchenne Muscular Dystrophy", 
                    "freq_in_cases": 0.45, "freq_in_controls": 0.00005},
    "rs128625268": {"gene": "DMD", "disease_name": "Duchenne Muscular Dystrophy", 
                    "freq_in_cases": 0.35, "freq_in_controls": 0.00004},
    "rs398123529": {"gene": "DMD", "disease_name": "Duchenne Muscular Dystrophy", 
                    "freq_in_cases": 0.20, "freq_in_controls": 0.00003},
    
    # Sickle Cell Disease (HBB gene)
    "rs334": {"gene": "HBB", "disease_name": "Sickle Cell Disease", 
              "freq_in_cases": 0.75, "freq_in_controls": 0.002},
    "rs33930165": {"gene": "HBB", "disease_name": "Sickle Cell Disease", 
                   "freq_in_cases": 0.15, "freq_in_controls": 0.0005},
    "rs35497102": {"gene": "HBB", "disease_name": "Sickle Cell Disease", 
                   "freq_in_cases": 0.10, "freq_in_controls": 0.0003},
    
    # Hereditary Breast Cancer (BRCA1/BRCA2)
    "rs80357906": {"gene": "BRCA1", "disease_name": "Hereditary Breast Cancer", 
                   "freq_in_cases": 0.35, "freq_in_controls": 0.001},
    "rs80356920": {"gene": "BRCA1", "disease_name": "Hereditary Breast Cancer", 
                   "freq_in_cases": 0.25, "freq_in_controls": 0.0008},
    "rs80359550": {"gene": "BRCA2", "disease_name": "Hereditary Breast Cancer", 
                   "freq_in_cases": 0.30, "freq_in_controls": 0.0015},
    "rs80358947": {"gene": "BRCA2", "disease_name": "Hereditary Breast Cancer", 
                   "freq_in_cases": 0.20, "freq_in_controls": 0.0006},
    
    # Lynch Syndrome (MLH1, MSH2, MSH6, PMS2)
    "rs63750217": {"gene": "MLH1", "disease_name": "Lynch Syndrome", 
                   "freq_in_cases": 0.30, "freq_in_controls": 0.002},
    "rs63750449": {"gene": "MSH2", "disease_name": "Lynch Syndrome", 
                   "freq_in_cases": 0.25, "freq_in_controls": 0.0018},
    "rs1064794302": {"gene": "MSH6", "disease_name": "Lynch Syndrome", 
                    "freq_in_cases": 0.20, "freq_in_controls": 0.0015},
    "rs121434629": {"gene": "PMS2", "disease_name": "Lynch Syndrome", 
                    "freq_in_cases": 0.15, "freq_in_controls": 0.001},
}

# Benign/common variants (not associated with diseases)
BENIGN_VARIANTS = {
    "rs1800054": {"gene": "ATM", "pop_freq": 0.15},
    "rs1800055": {"gene": "ATM", "pop_freq": 0.12},
    "rs1800056": {"gene": "ATM", "pop_freq": 0.10},
    "rs1042522": {"gene": "TP53", "pop_freq": 0.25},
    "rs17878362": {"gene": "TP53", "pop_freq": 0.02},
    "rs1625895": {"gene": "TP53", "pop_freq": 0.18},
    "rs4986850": {"gene": "BRCA1", "pop_freq": 0.08},
    "rs799917": {"gene": "BRCA1", "pop_freq": 0.30},
    "rs16942": {"gene": "BRCA1", "pop_freq": 0.28},
    "rs169547": {"gene": "BRCA2", "pop_freq": 0.22},
}


def generate_patient_cohort(node_id: str, total_patients: int = 500, regional_variation: float = 0.1):
    """
    Generate an enriched clinical cohort for a specific node
    
    Args:
        node_id: Identifier for the node
        total_patients: Total number of patients per disease comparison
        regional_variation: Factor for regional frequency variations
    
    Returns:
        DataFrame with patient-level data
    """
    
    patients = []
    patient_id_counter = 1
    
    # Add regional variation
    np.random.seed(hash(node_id) % 2**32)  # Different seed per node
    regional_factor = 1 + np.random.uniform(-regional_variation, regional_variation, size=len(CASES_PER_DISEASE))
    
    """
        For example: node_id = "node1", total_patients = 500 and regional_variation = 0.1
        Then regional_factor might be something like [1.05, 0.95, 1.10, 0.90, 1.00, 1.08]
        CASES_PER_DISEASE = 6 diseases
        
    """
    names_header = ["name", "full_name", "patient_name", "subject_name"]
    age_header = ["age", "Age", "patient_age", "subject_age"]
    height_header = ["height_cm", "Height", "patient_height", "subject_height"]
    weight_header = ["weight_kg", "Weight", "patient_weight", "subject_weight"]
    marrital_status_header = ["marital_status", "MaritalStatus", "patient_marital_status", "subject_marital_status"]
    id_header = ["PatientID", "SUBJ_NO", "pid", "Patient_Identifier"]
    disease_header = ["disease", "disease_name", "diagnosis", "condition"]
    condition_header = ["disease_condition", "disease_status", "status", "diagnosis_status"]
    sex_header = ["sex", "Sex", "gender", "Gender"]
    int_node_ID = int(node_id.replace("node", ""))
    
    for i, (disease, base_n_cases) in enumerate(CASES_PER_DISEASE.items()):
        
        # Adjust case numbers slightly for regional variation
        n_cases = max(5, int(base_n_cases * regional_factor[i]))
        n_controls = total_patients - n_cases

        # Get variants associated with this disease
        disease_variants = {k: v for k, v in PATHOGENIC_VARIANTS.items() if v["disease_name"] == disease}
        
        # Generate cases (patients with the disease)
        for _ in range(n_cases):
            name, age, sex, height, weight, marital_status = generate_synthetic_personal_data()
            case_int_node_ID = int_node_ID - 1
            patient = {
                # Options to include personal data for cases
                names_header[case_int_node_ID]: name,
                age_header[case_int_node_ID]: age,
                sex_header[case_int_node_ID]: sex,
                height_header[case_int_node_ID]: height,
                weight_header[case_int_node_ID]: weight,
                marrital_status_header[case_int_node_ID]: marital_status,

                id_header[case_int_node_ID]: f"{node_id}_P{patient_id_counter:04d}",
                "node_id": node_id,
                disease_header[case_int_node_ID]: disease,
                condition_header[case_int_node_ID]: True
            }
            
            # For cases, use the case frequency for pathogenic variants
            for variant_id, variant_info in disease_variants.items():
                patient[variant_id] = np.random.random() < variant_info["freq_in_cases"]
            
            # Add benign variants with population frequency
            for variant_id, variant_info in BENIGN_VARIANTS.items():
                patient[variant_id] = np.random.random() < variant_info["pop_freq"]
            
            # Add other disease variants at control frequency
            for variant_id, variant_info in PATHOGENIC_VARIANTS.items():
                if variant_info["disease_name"] != disease and variant_id not in patient:
                    patient[variant_id] = np.random.random() < variant_info.get("freq_in_controls", 0.001)
            
            patients.append(patient)
            patient_id_counter += 1
        
        # Generate controls (patients without the disease)
        for _ in range(n_controls):
            name, age, sex, height, weight, marital_status = generate_synthetic_personal_data()
            control_int_node_ID = int_node_ID - 1
            patient = {
                # Options to include personal data for controls
                names_header[control_int_node_ID]: name,
                age_header[control_int_node_ID]: age,
                sex_header[control_int_node_ID]: sex,
                height_header[control_int_node_ID]: height,
                weight_header[control_int_node_ID]: weight,
                marrital_status_header[control_int_node_ID]: marital_status,

                id_header[control_int_node_ID]: f"{node_id}_P{patient_id_counter:04d}",
                "node_id": node_id,
                disease_header[control_int_node_ID]: disease,
                condition_header[control_int_node_ID]: False
            }
            
            # For controls, use control frequency for all pathogenic variants
            for variant_id, variant_info in PATHOGENIC_VARIANTS.items():
                patient[variant_id] = np.random.random() < variant_info.get("freq_in_controls", 0.001)
            
            # Add benign variants with population frequency
            for variant_id, variant_info in BENIGN_VARIANTS.items():
                patient[variant_id] = np.random.random() < variant_info["pop_freq"]
            
            patients.append(patient)
            patient_id_counter += 1
    
    # Create DataFrame
    df = pd.DataFrame(patients)
    
    # Fill NaN values with False (variant not present)
    variant_columns = list(PATHOGENIC_VARIANTS.keys()) + list(BENIGN_VARIANTS.keys())
    for col in variant_columns:
        if col in df.columns:
            df[col] = df[col].fillna(False).infer_objects(copy=False).astype(bool)
    
    return df

def generate_variant_metadata():
    """Generate metadata file for variants"""
    metadata = {}
    
    for variant_id, info in PATHOGENIC_VARIANTS.items():
        # Calculate odds ratio from frequencies
        freq_cases = info.get("freq_in_cases", 0.5)
        freq_controls = info.get("freq_in_controls", 0.001)
        
        # Prevent division by zero (What formula is used?)
        if freq_controls > 0 and freq_controls < 1 and freq_cases < 1:
            odds_cases = freq_cases / (1 - freq_cases)
            odds_controls = freq_controls / (1 - freq_controls)
            odds_ratio = odds_cases / odds_controls if odds_controls > 0 else 999
        else:
            odds_ratio = 999 if freq_cases > freq_controls else 1
        
        metadata[variant_id] = {
            "gene": info["gene"],
            "associated_disease": info["disease_name"],
            "pathogenicity": "Pathogenic",
            "odds_ratio": round(odds_ratio, 2),
            "frequency_in_cases": freq_cases,
            "frequency_in_controls": freq_controls
        }

    for variant_id, info in BENIGN_VARIANTS.items():
        metadata[variant_id] = {
            "gene": info["gene"],
            "associated_disease": None,
            "pathogenicity": "Benign",
            "odds_ratio": 1.0,
            "population_frequency": info["pop_freq"]
        }
    
    return metadata


def main():
    """Generate patient data for all 4 nodes"""
    
    print("Generating enriched clinical cohorts for rare disease analysis...")
    print(f"Each node will have 500 patients per disease (enriched cohorts)")
    print(f"Diseases included: {', '.join(CASES_PER_DISEASE.keys())}")
    print()
    
    # Generate variant metadata
    metadata = generate_variant_metadata()
    with open("variant_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("✓ Generated variant_metadata.json")
    
    # Generate data for each node with slight regional variations
    regional_variations = [0.0, 0.1, 0.15, 0.2]  # Different regional variations
    
    for i in range(1, 5):
        node_id = f"node{i}"
        df = generate_patient_cohort(node_id, total_patients=500, 
                                    regional_variation=regional_variations[i-1])
    
        # Ensure output directory exists
        import os
        os.makedirs(f"nodes/{node_id}", exist_ok=True)
        
        filename = f"nodes/{node_id}/patients_{node_id}"
        df.to_csv(f"{filename}.csv", index=False) # Files are saved in CSV format
        
        # Options to save in other formats:
        # df.to_csv(f"{filename}.tsv", sep='\t', index=False) # Files are saved in TSV format
        # df.index = range(1, len(df) + 1) # Indexing will start at index 1 for JSON output
        # df.to_json(f"{filename}.json", orient='index', indent=3) # Files are saved in JSON format
        
        # Print summary statistics
        print(f"\n✓ Generated {filename} with all format (CSV, TSV, JSON):")
        print(f"  Total patients: {len(df)}")
        print(f"  Unique patients: {df[df.columns[0]].nunique()}")
        print(f"  Cases by disease:")
        
        for disease in CASES_PER_DISEASE.keys():
            disease_df = df[df[df.columns[2]] == disease]
            n_case = disease_df[disease_df.columns[3]].sum()
            n_control = len(disease_df) - n_case
            print(f"    {disease}: {n_case} cases, {n_control} controls")
        
        # Show example Fisher's test power for a key variant
        if i == 1:  # Only show for first node
            print(f"\n  Example variant frequencies (node1):")
            cf_data = df[df[df.columns[2]] == 'Cystic Fibrosis']
            if 'rs113993960' in cf_data.columns:
                cf_cases = cf_data[cf_data[disease_df.columns[3]] == True]
                cf_controls = cf_data[cf_data[disease_df.columns[3]] == False]
                var_in_cases = cf_cases['rs113993960'].sum() if len(cf_cases) > 0 else 0
                var_in_controls = cf_controls['rs113993960'].sum() if len(cf_controls) > 0 else 0
                print(f"    CFTR rs113993960 in CF: {var_in_cases}/{len(cf_cases)} cases, "
                      f"{var_in_controls}/{len(cf_controls)} controls")
    
    print("\n✓ Enriched clinical cohort generation complete!")
    print("\nNote: These represent specialized clinical centers with enriched")
    print("      rare disease cohorts, not general population samples.")
    print("\nFiles created:")
    print("  - variant_metadata.json (variant information)")
    print("  - patients_node1.csv (3000 patients)")
    print("  - patients_node2.csv (3000 patients)")
    print("  - patients_node3.csv (3000 patients)")
    print("  - patients_node4.csv (3000 patients)")

if __name__ == "__main__":
    main()
