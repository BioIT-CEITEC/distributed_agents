
import pandas as pd
import logging
from schema_manager import SchemaManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifySchema")

def run_verification():
    schema_manager = SchemaManager(logger=logger)
    
    # Test Case 1: Standard Messy Headers
    print("\n--- Test Case 1: Loose Matching ---")
    df1 = pd.DataFrame({
        'My_Patient_ID_Number': [1, 2],
        'Primary_Disease_Name': ['Flu', 'Cold'],
        'Current_Disease_Status': ['Active', 'Recovered'],
        'Patient_Age': [30, 40], # Should be dropped? Or kept if matched by chance? Actually age is not in safe list unless matched.
        'Patient_Name': ['John', 'Jane'] # Should be dropped (PII)
    })
    
    normalized_df1 = schema_manager.normalize_dataframe(df1)
    print("Original Columns:", df1.columns.tolist())
    print("Normalized Columns:", normalized_df1.columns.tolist())
    
    # Assertions
    assert 'patient_id' in normalized_df1.columns, "Failed to map patient_id"
    assert 'disease' in normalized_df1.columns, "Failed to map disease"
    assert 'has_disease' in normalized_df1.columns, "Failed to map has_disease"
    assert 'Patient_Name' not in normalized_df1.columns, "Failed to drop sensitive data"
    
    # Test Case 2: Collision Avoidance
    print("\n--- Test Case 2: Collision Avoidance (Disease vs Status) ---")
    df2 = pd.DataFrame({
        'Participant_ID': [1],
        'Condition': ['Diabetes'],          # Should map to 'disease'
        'Condition_Status': ['Confirmed'],  # Should map to 'has_disease' - verify 'Condition' doesn't greedily match this? 
                                            # Actually 'Condition_Status' matches 'condition_?status' pattern.
                                            # But does 'Condition' regex match 'Condition_Status'? 
                                            # The new regex for disease is `(?i)^.*condition(?!.*status).*$`. 
                                            # So 'Condition_Status' should NOT match the disease pattern.
    })
    
    normalized_df2 = schema_manager.normalize_dataframe(df2)
    print("Original Columns:", df2.columns.tolist())
    print("Normalized Columns:", normalized_df2.columns.tolist())
    
    assert 'disease' in normalized_df2.columns
    assert 'has_disease' in normalized_df2.columns
    
    # Check that we didn't lose data
    print("Data:\n", normalized_df2)

    print("\nVERIFICATION SUCCESSFUL")

if __name__ == "__main__":
    run_verification()
