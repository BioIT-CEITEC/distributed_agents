import numpy as np
import random

def generate_synthetic_personal_data():
    """
    Generates a synthetic dataset of personal data with realistic distributions.
    """
    
    # Realistic names from diverse regions
    first_names_male = [
        'John', 'James', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Christopher',
        'Carlos', 'Jose', 'Juan', 'Miguel', 'Luis', 'Antonio', 'Francisco', 'Manuel', 'Javier', 'Daniel',
        'Ahmed', 'Mohammed', 'Ali', 'Omar', 'Hassan', 'Hussein', 'Youssef', 'Ibrahim', 'Abdullah', 'Khalid',
        'Rajesh', 'Amit', 'Vikram', 'Arjun', 'Ravi', 'Sanjay', 'Deepak', 'Prakash', 'Sunil', 'Ramesh',
        'Wei', 'Ming', 'Hao', 'Jian', 'Lei', 'Peng', ' Qiang', 'Tao', 'Xiao', 'Zhang',
        'Pierre', 'Jean', 'Michel', 'Philippe', 'Patrick', 'Bernard', 'Alain', 'Jacques', 'Pascal', 'Laurent',
        'Kwame', 'Kofi', 'Amin', 'Juma', 'Sule', 'Babatunde', 'Chukwu', 'Omar', 'Idris', ' Musa'
    ]

    first_names_female = [
        'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen',
        'Maria', 'Carmen', 'Ana', 'Isabel', 'Sofia', 'Lucia', 'Rosa', 'Teresa', 'Pilar', 'Dolores',
        'Fatima', 'Aisha', 'Zainab', 'Maryam', 'Khadija', 'Sara', 'Huda', 'Laila', 'Nadia', 'Rania',
        'Priya', 'Asha', 'Lakshmi', 'Sunita', 'Meera', 'Ritu', 'Pooja', 'Neha', 'Divya', 'Kavita',
        'Li', 'Mei', 'Xia', 'Jing', 'Yan', 'Fen', 'Lan', 'Hua', 'Ping', 'Qiu',
        'Marie', 'Sophie', 'Claire', 'Julie', 'Nathalie', 'Isabelle', 'Christine', 'Anne', 'Valerie', 'Brigitte',
        'Fatou', 'Aminata', 'Awa', 'Mariama', 'Ndoye', 'Sokhna', 'Maimouna', 'Penda', 'Adama', 'Ndeye'
    ]

    last_names = [
        'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
        'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
        'Khan', 'Patel', 'Singh', 'Ali', 'Wang', 'Li', 'Zhang', 'Chen', 'Liu', 'Yang',
        'Al-Fayed', 'Omar', 'Hussein', 'Kumar', 'Sharma', 'Gupta', 'Rao', 'Nair', 'Silva', 'Santos',
        'Dupont', 'Martin', 'Bernard', 'Dubois', 'Moreau', 'Fischer', 'Muller', 'Schmitt', 'Weber', 'Meyer',
        'Nkosi', 'Mthembu', 'Zulu', 'Ndlovu', 'Khoza', 'Dlamini', 'Mbatha', 'Sithole', 'Mabaso', 'Ngubane'
    ]

    marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
    sexes = ['Male', 'Female']
    
    sex = random.choice(sexes)
    first_name = random.choice(first_names_male if sex == 'Male' else first_names_female)
    last_name = random.choice(last_names)
    name = f"{first_name} {last_name}"
    age = np.random.randint(22, 81)
    
    # Realistic height/weight correlations by sex/age
    if sex == 'Male':
        height = np.random.normal(175, 8, 1)[0]  # cm
        weight = np.random.normal(80, 15, 1)[0]  # kg
    else:
        height = np.random.normal(162, 7, 1)[0]
        weight = np.random.normal(65, 12, 1)[0]
    
    height = max(140, min(210, round(height)))  # Clamp realistic range
    weight = max(40, min(150, round(weight)))
    
    marital_status = random.choice(marital_statuses)

    return name, age, sex, height, weight, marital_status
