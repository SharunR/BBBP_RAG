import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import KMeansSMOTE
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

df=pd.read_csv(r"C:\Edu\project\latelkimages\BBBP.csv")
def extract_features(smiles_list):
    fp_gen = AllChem.GetMorganGenerator(radius=2, fpSize=128)
    descriptor_names = [desc[0] for desc in Descriptors._descList]

    features = []
    valid_indices = []

    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp_vect = fp_gen.GetFingerprint(mol)
            fp_array = np.zeros((0,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp_vect, fp_array)

            desc_array = []
            for name in descriptor_names:
                func = getattr(Descriptors, name)
                desc_array.append(func(mol))

            combined = np.concatenate([fp_array, np.array(desc_array)])
            features.append(combined)
            valid_indices.append(idx)

    return np.array(features), valid_indices


raw_features, valid_idx = extract_features(list(df['smiles']))
y = df['p_np'].iloc[valid_idx].values

X_train, X_test, y_train, y_test = train_test_split(raw_features, y, test_size=0.2, random_state=42, stratify=y)

X_train = np.nan_to_num(X_train, nan=0.0)
X_train = np.clip(X_train, -1e6, 1e6)

X_test = np.nan_to_num(X_test, nan=0.0)
X_test = np.clip(X_test, -1e6, 1e6)

kmeans_smote = KMeansSMOTE(random_state=42, cluster_balance_threshold=0.1)
X_train_res, y_train_res = kmeans_smote.fit_resample(X_train, y_train)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train_res, y_train_res)
preds = model.predict(X_test)
import joblib

joblib.dump(model,"bbbp_xgb.pkl")
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds))