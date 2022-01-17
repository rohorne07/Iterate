from .jtnnencoder import JTNNEmbed

#%% Test functioning
smiles = ['C1=CC=C(C=C1)NC2=CC(=CC=C2)O']
jtnn = JTNNEmbed(smiles)

features = jtnn.get_features()

features


#%% Embed dataframe of smiles
import pandas as pd

df = pd.read_csv('../data/zinc.csv')

df.head()
df.shape
lst = list(df['SMILES'])

#process all smiles and featurise


smiles = []
representation = []
errormol = []

for entry in lst:
    try:
        smiles.append(entry)
        jtnn = JTNNEmbed([entry])
        features = jtnn.get_features()
        representation.append(features)
    except Exception:
        errormol.append(entry)
        representation.append('ERROR')
        continue


df['Feature'] = representation

#note the other functions require commma separated values to process once read from a dataframe

