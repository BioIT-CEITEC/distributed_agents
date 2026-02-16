import json
import random
import csv
import os
from pathlib import Path

# Configuration
NODES = 4
PATIENTS_PER_NODE = 3000
BLOCK_SIZE = 500
OUTPUT_FILES = {
    1: "nodes/node1/czech_patient.csv",
    2: "nodes/node2/berlin_patient.csv",
    3: "nodes/node3/london_patient.csv",
    4: "nodes/node4/paris_patient.csv"
}

# Distinct Headers per Node
HEADERS = {
    1: {
        "id": "patient_id", 
        "node": "node_id", 
        "disease": "disease_name", 
        "status": "has_disease", 
        "variants": "variants_id"
    },
    2: {
        "id": "pid", 
        "node": "nid", 
        "disease": "condition", 
        "status": "diagnosis_status", 
        "variants": "genotypes"
    },
    3: {
        "id": "subject_id", 
        "node": "site_id", 
        "disease": "diagnosis", 
        "status": "affected", 
        "variants": "mutations"
    },
    4: {
        "id": "participant_id", 
        "node": "center_id", 
        "disease": "medical_condition", 
        "status": "status", 
        "variants": "snps"
    }
}

def load_metadata():
    with open("variant_metadata.json", "r") as f:
        return json.load(f)

def generate_patient_block(disease_name, variants_data, num_patients, node_id):
    """Generates a block of patients for a specific disease"""
    
    # Identify associated variants for this disease
    associated_variants = [
        v_id for v_id, info in variants_data.items() 
        if info.get("associated_disease") == disease_name
    ]
    
    other_variants = [
        v_id for v_id, info in variants_data.items() 
        if info.get("associated_disease") != disease_name
    ]
    
    patients = []
    
    # Randomize case ratio for this block (between 30% and 40% cases)
    case_ratio = random.uniform(0.30, 0.40)
    
    for _ in range(num_patients):
        # Determine status
        is_case = random.random() < case_ratio
        
        patient_variants = set()
        
        # 1. Add associated variants based on status-specific frequency
        for v_id in associated_variants:
            if is_case:
                freq = variants_data[v_id].get("frequency_in_cases", 0.0)
            else:
                freq = variants_data[v_id].get("frequency_in_controls", 0.001)
                
            if random.random() < freq:
                patient_variants.add(v_id)
                
        # 2. Add random other variants (background noise)
        # Ensure total variants per patient is between 3 and 10
        target_count = random.randint(3, 10)
        
        while len(patient_variants) < target_count:
            if not other_variants:
                 break
            v_id = random.choice(other_variants)
            # Use control frequency for non-associated variants
            freq = variants_data[v_id].get("frequency_in_controls", 0.01) 
            # Boost frequency to ensure we find variants reasonably fast
            if random.random() < max(freq * 10, 0.2): 
                patient_variants.add(v_id)
                
        patients.append({
            "disease_real": disease_name,
            "has_disease_real": is_case,
            "variants_set": patient_variants
        })
        
    return patients

def main():
    print("Loading variant metadata...")
    try:
        metadata = load_metadata()
    except FileNotFoundError:
        print("Error: variant_metadata.json not found.")
        return

    # Get disease list from metadata
    diseases = sorted(list(set(
        info["associated_disease"] 
        for info in metadata.values() 
        if info.get("associated_disease") is not None
    )))
    
    print(f"Found {len(diseases)} diseases: {diseases}")
    
    for node_id in range(1, NODES + 1):
        filename = OUTPUT_FILES[node_id]
        print(f"Generating data for Node {node_id} ({filename})...")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        node_patients = []
        patient_counter = 1
        
        # Generate blocks
        for disease in diseases:
            block = generate_patient_block(disease, metadata, BLOCK_SIZE, node_id)
            
            for p in block:
                # Basic info
                p_id_str = f"node{node_id}_P{patient_counter:04d}"
                
                # Create row with node-specific headers
                # Use a consistent order for CSV: id, node, disease, status, variants
                row = {
                    HEADERS[node_id]["id"]: p_id_str,
                    HEADERS[node_id]["node"]: f"node{node_id}",
                    HEADERS[node_id]["disease"]: p["disease_real"],
                    HEADERS[node_id]["status"]: p["has_disease_real"],
                    HEADERS[node_id]["variants"]: ";".join(sorted(list(p["variants_set"])))
                }
                    
                node_patients.append(row)
                patient_counter += 1
                
        # Shuffle (simulating random registry order)
        random.shuffle(node_patients)
        
        # Save to CSV
        if node_patients:
            fieldnames = list(node_patients[0].keys())
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(node_patients)
                
            print(f"  -> Saved {len(node_patients)} records to {filename}")

if __name__ == "__main__":
    main()