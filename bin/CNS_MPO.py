from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors

class SmilesError(Exception): pass

def log_partition_coefficient(smiles):
    '''
    Returns the octanol-water partition coefficient given a molecule SMILES 
    string
    '''
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        raise SmilesError('%s returns a None molecule' % smiles)
        
    return Crippen.MolLogP(mol)
    
def lipinski_trial(smiles):
    '''
    Returns which of Lipinski's rules a molecule has failed, or an empty list
    
    Lipinski's rules are:
    Hydrogen bond donors <= 5
    Hydrogen bond acceptors <= 10
    Molecular weight < 500 daltons
    logP < 5
    '''
    passed = []
    failed = []
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Exception('%s is not a valid SMILES string' % smiles)
    
    num_hdonors = Lipinski.NumHDonors(mol)
    num_hacceptors = Lipinski.NumHAcceptors(mol)
    mol_weight = Descriptors.MolWt(mol)
    mol_logp = Crippen.MolLogP(mol)
    
    failed = []
    
    if num_hdonors > 5:
        failed.append('Over 5 H-bond donors, found %s' % num_hdonors)
    else:
        passed.append('Found %s H-bond donors' % num_hdonors)
        
    if num_hacceptors > 10:
        failed.append('Over 10 H-bond acceptors, found %s' \
        % num_hacceptors)
    else:
        passed.append('Found %s H-bond acceptors' % num_hacceptors)
        
    if mol_weight >= 500:
        failed.append('Molecular weight over 500, calculated %s'\
        % mol_weight)
    else:
        passed.append('Molecular weight: %s' % mol_weight)
        
    if mol_logp >= 5:
        failed.append('Log partition coefficient over 5, calculated %s' \
        % mol_logp)
    else:
        passed.append('Log partition coefficient: %s' % mol_logp)
    
    return passed, failed
    
def lipinski_pass(smiles):
    '''
    Wraps around lipinski trial, but returns a simple pass/fail True/False
    '''
    passed, failed = lipinski_trial(smiles)
    if failed:
        return False
    else:
        return True


def cns_mpo(value, entry_type=None):
    if entry_type == 'Mol Weight':
        if value <= 360.0: return 1.0
        elif value >= 500.0: return 0.0
        else: return -1./140.*(value-500.0)
    elif entry_type == 'LogD':
        if value <= 2.0: return 1.0
        elif value >= 4.0: return 0.0
        else: return -0.5*value+2.0
    elif entry_type == 'LogP':
        if value <= 3.0: return 1.0
        elif value >= 5.0: return 0.0
        else: return -0.5*(value-5.0)
    elif entry_type == 'Strongest basic pKa':
        if value <= 8.0: return 1.0
        elif value >= 10.0: return 0.0
        else: return -0.5*value+5.0
    elif entry_type == 'TPSA':
        if value <= 20 or value >= 120: return 0.0
        elif value >=40 and value <= 90: return 1.0
        elif value > 20 and value <40: return 1./20.*(value-20.0) 
        else: return -1./30.*(value-120.0) 
    elif entry_type == 'H bond donors':
        if value <= 0.5: return 1.0  
        elif value >= 3.5: return 0.0
        else: return -1./3.*(value-3.5)
    else:
        print("Warning: wrong field in the CNS_MPO calculations")
        return None

ROOT = "../data/zinc.csv"

import pandas as pd

df = pd.read_csv(ROOT)
df = df[~df['Mol Weight'].str.startswith('Error')]

df['MPO_MW'] = df['Mol Weight'].apply(float).apply(cns_mpo, args=('Mol Weight',))
df['MPO_logD'] = df['LogD'].apply(float).apply(cns_mpo, args=('LogD',))
df['MPO_logP'] = df['LogP'].apply(float).apply(cns_mpo, args=('LogP',))
df['MPO_basicpKa'] = df['Basic pKa'].apply(float).apply(cns_mpo, args=('Strongest basic pKa',))
df['MPO_TPSA'] = df['TPSA'].apply(float).apply(cns_mpo, args=('TPSA',))
df['MPO_HBD'] = df['HBD'].apply(float).apply(cns_mpo, args=('H bond donors',))
df['CNS_MPO'] = df[['MPO_MW', 'MPO_logD', 'MPO_logP', 'MPO_basicpKa', 'MPO_TPSA', 'MPO_HBD']].sum(axis=1)

df.to_csv(ROOT+'_MPO2.csv')
