"""
TRAIN MW-SPLIT DENSITY DICTIONARIES

Comparison experiment:

Model A:
    Non-Negative Least Squares (NNLS)

    ΔV >= 0


Model B:
    Ordinary Least Squares (OLS)

    ΔV unrestricted


Both models use:
    V_L = C * Σ(m_i^(1/3))

and:

    V_pred = V_L - Σ(ΔV_b n_b)


Purpose:
Determine whether the physical non-negativity
constraint improves or reduces predictive accuracy.
"""


import numpy as np
import pandas as pd

from rdkit import Chem

from scipy.optimize import nnls

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# CONSTANTS
# ============================================================


PHI = (1 + np.sqrt(5)) / 2
MAGIC_SCALE = np.pi * np.pi / PHI


MW_SMALL_MAX = 180
MW_MEDIUM_MAX = 400


DEFAULT_OVERLAP = 5.9


# ============================================================
# DESCRIPTORS
# ============================================================


def get_molecular_weight(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    return sum(
        atom.GetMass()
        for atom in mol.GetAtoms()
    )



def magic_volume(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    return MAGIC_SCALE * sum(
        atom.GetMass() ** (1/3)
        for atom in mol.GetAtoms()
    )



def get_bond_types(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None


    mol = Chem.AddHs(mol)

    bonds=[]


    for bond in mol.GetBonds():

        a1=bond.GetBeginAtom().GetSymbol()
        a2=bond.GetEndAtom().GetSymbol()


        order = int(
            bond.GetBondTypeAsDouble()
        )


        if a1>a2:
            a1,a2=a2,a1


        bonds.append(
            (
                a1,
                a2,
                order
            )
        )


    return bonds



# ============================================================
# LOAD DATA
# ============================================================


print("="*80)
print("LOADING DATA")
print("="*80)


df=pd.read_csv(
    "takuyhaa.csv"
)


df=df.dropna(
    subset=[
        "SMILES",
        "Density"
    ]
)



data=[]


for _,row in tqdm(
    df.iterrows(),
    total=len(df)
):

    smiles=row["SMILES"]


    mw=get_molecular_weight(smiles)

    V=magic_volume(smiles)

    bonds=get_bond_types(smiles)


    if (
        mw is None or
        V is None or
        bonds is None
    ):
        continue



    data.append(
        {
        "SMILES":smiles,
        "MW":mw,
        "V_magic":V,
        "V_exp":mw/row["Density"],
        "Density":row["Density"],
        "bonds":bonds
        }
    )


print()
print("Valid molecules:",len(data))



# ============================================================
# TRAIN DICTIONARY
# ============================================================


def train_dictionary(
    dataset,
    mode="nnls",
    min_count=0
):


    bond_frequency={}


    for row in dataset:

        for b in row["bonds"]:

            bond_frequency[b]=(
                bond_frequency.get(b,0)+1
            )



    bond_types=[
        b for b,c in bond_frequency.items()
        if c>=min_count
    ]



    X=[]
    y=[]


    for row in dataset:


        counts={
            b:0
            for b in bond_types
        }


        for b in row["bonds"]:

            if b in counts:
                counts[b]+=1



        X.append(
            [
                counts[b]
                for b in bond_types
            ]
        )


        y.append(
            row["V_magic"] -
            row["V_exp"]
        )



    X=np.array(X)
    y=np.array(y)



    if mode=="nnls":

        coeff,_=nnls(
            X,
            y
        )


    elif mode=="ols":

        model=LinearRegression(
            fit_intercept=False
        )

        model.fit(
            X,
            y
        )

        coeff=model.coef_



    dictionary=dict(
        zip(
            bond_types,
            coeff
        )
    )


    return dictionary



# ============================================================
# PREDICTION
# ============================================================


def predict_density(
    row,
    dictionary
):


    correction=0


    for b in row["bonds"]:

        correction += dictionary.get(
            b,
            DEFAULT_OVERLAP
        )



    V_corrected=(
        row["V_magic"]
        -
        correction
    )


    if V_corrected<=0:

        V_corrected=row["V_magic"]



    return (
        row["MW"]
        /
        V_corrected
    )



# ============================================================
# RUN BOTH MODELS
# ============================================================


print()
print("="*80)
print("TRAINING DICTIONARIES")
print("="*80)


nnls_dict=train_dictionary(
    data,
    mode="nnls"
)


ols_dict=train_dictionary(
    data,
    mode="ols"
)



print(
"NNLS bonds:",
len(nnls_dict)
)

print(
"OLS bonds:",
len(ols_dict)
)



# ============================================================
# EVALUATION
# ============================================================


results=[]


for name,dictionary in [

    ("NNLS",nnls_dict),

    ("OLS unrestricted",ols_dict)

]:


    predictions=[]

    actual=[]



    for row in data:

        predictions.append(
            predict_density(
                row,
                dictionary
            )
        )

        actual.append(
            row["Density"]
        )



    predictions=np.array(predictions)
    actual=np.array(actual)



    mae=mean_absolute_error(
        actual,
        predictions
    )


    rmse=np.sqrt(
        np.mean(
            (actual-predictions)**2
        )
    )


    r2=r2_score(
        actual,
        predictions
    )


    results.append(
        {
        "Model":name,
        "MAE":mae,
        "RMSE":rmse,
        "R2":r2
        }
    )


summary=pd.DataFrame(results)



print()
print("="*80)
print("FINAL COMPARISON")
print("="*80)


print(
summary.to_string(
    index=False
)
)



summary.to_csv(
    "NNLS_vs_OLS_dictionary_comparison.csv",
    index=False
)


print()
print(
"Saved NNLS_vs_OLS_dictionary_comparison.csv"
)