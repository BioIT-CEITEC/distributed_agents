import pandas as pd
import re
import logging
from typing import List, Dict, Optional, Tuple

class SchemaManager:
    """
    Intelligently infers and maps heterogeneous CSV headers to a standard schema.
    Also enforces privacy by detecting and removing sensitive columns.
    """
    
    # Standard internal column names
    STD_PATIENT_ID = "patient_id"
    STD_DISEASE = "disease"
    STD_HAS_DISEASE = "has_disease"
    
    # Common variations for distinct clinical fields
    _PATIENT_ID_PATTERNS = [
        r"(?i)^.*patient.*id.*$", 
        r"(?i)^.*subj(ect)?.*id.*$", 
        r"(?i)^.*participant.*id.*$", 
        r"(?i)^.*sample.*id.*$",
        r"(?i)^.*pid.*$"
    ]
    
    _DISEASE_NAME_PATTERNS = [
        r"(?i)^.*disease(?!.*status)(?!.*condition).*$", # Modified to avoid overlap if 'disease_condition' is status 
        r"(?i)^.*condition(?!.*status).*$", 
        r"(?i)^.*diagnosis(?!.*status).*$", 
        r"(?i)^.*phenotype.*$",
        r"(?i)^.*indication.*$",
        r"(?i)^.*diagnosis_?name.*$",
        r"(?i)^.*disorder(?!.*status).*$",
        r"(?i)^.*illness(?!.*status).*$",
        r"(?i)^.*disease_?term.*$"
    ]
    
    _DISEASE_STATUS_PATTERNS = [
        r"(?i)^.*has_?disease.*$", 
        r"(?i)^.*status.*$", 
        r"(?i)^.*disease_?status.*$", 
        r"(?i)^.*affected.*$", 
        r"(?i)^.*diagnosis_?status.*$", 
        r"(?i)^.*condition_?status.*$",
        r"(?i)^.*class.*$",
        r"(?i)^.*disease_?condition.*$",
        r"(?i)^.*health_?condition.*$",
        r"(?i)^.*medical_?condition.*$"
    ]
    
    # Sex/Gender (to be preserved)
    _SEX_PATTERNS = [r"(?i)^sex$", r"(?i)^gender$"]
    
    # Variant pattern (rsID)
    _VARIANT_PATTERN = r"(?i)^rs\d+$"
    
    # SENSITIVE DATA PATTERNS (PII) - BLACKLIST
    # Columns matching these will be explicitly dropped
    _SENSITIVE_PATTERNS = [
        r"(?i).*name.*",          # name, first_name, last_name, full_name
        r"(?i).*address.*",       # address, home_address, email_address
        r"(?i).*email.*",         # email, contact_email
        r"(?i).*phone.*",         # phone, mobile, contact_number
        r"(?i).*ssn.*",           # ssn, social_security
        r"(?i).*dob.*",           # dob, date_of_birth (often considered sensitive in small cohorts)
        r"(?i).*birth.*",         # birth_date, birthday
        r"(?i).*zip.*",           # zip_code, postal_code
        r"(?i).*contact.*",       # contact_info
    ]

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("SchemaManager")

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a raw DataFrame with unknown headers and:
        1. Identifies and renames core columns (patient_id, disease, etc.)
        2. Identifies variant columns
        3. DROPS all other columns (implicit whitelist) or drops known sensitive ones.
        
        Using a 'whitelist' approach is safer for privacy:
        We keep ONLY what we recognize as clinical data:
        - Mapped core columns
        - Variant columns (rsIDs)
        - Basic demographics if matched (Sex/Age - optional, let's keep Sex as explicitly safe)
        Everything else is discarded.
        """
        
        original_cols = df.columns.tolist()
        mapping = {}
        
        # 1. Identify Core Columns
        mapping[self._find_match(original_cols, self._PATIENT_ID_PATTERNS)] = self.STD_PATIENT_ID
        mapping[self._find_match(original_cols, self._DISEASE_NAME_PATTERNS)] = self.STD_DISEASE
        mapping[self._find_match(original_cols, self._DISEASE_STATUS_PATTERNS)] = self.STD_HAS_DISEASE
        
        # Optional: Map Sex if found
        sex_col = self._find_match(original_cols, self._SEX_PATTERNS)
        if sex_col:
            mapping[sex_col] = 'sex'
            
        # Remove None matches
        mapping = {k: v for k, v in mapping.items() if k is not None}
        
        # 2. Rename columns
        df_renamed = df.rename(columns=mapping)
        
        # 3. Identify Safe Columns to KEEP
        # Safe = Mapped columns + Variant columns (rs...)
        safe_columns = list(mapping.values())
        
        current_cols = df_renamed.columns.tolist()
        variant_cols = [c for c in current_cols if re.match(self._VARIANT_PATTERN, c)]
        
        # Add variants to safe list
        safe_columns.extend(variant_cols)
        
        # 4. Filter DataFrame (The Privacy Step)
        # Only keep columns that are in our safe list
        # This implementation effectively drops "name", "address", "random_notes", etc.
        final_df = df_renamed[safe_columns].copy()
        
        # Log what we did
        dropped_cols = [c for c in original_cols if c not in mapping and c not in variant_cols]
        if dropped_cols:
            self.logger.info(f"SchemaManager: Dropped {len(dropped_cols)} columns for privacy/relevance: {dropped_cols[:5]}...")
        
        self.logger.info(f"SchemaManager: Mapped {mapping}")
        
        return final_df

    def _find_match(self, columns: List[str], patterns: List[str]) -> Optional[str]:
        """Finds the first column that matches any of the given regex patterns."""
        for pattern in patterns:
            for col in columns:
                if re.match(pattern, col):
                    return col
        return None
