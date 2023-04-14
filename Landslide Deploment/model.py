import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import wandb
import pickle

wandb.login()
wandb.init(project='landslide-prediction',entity='KuenSolanin', name='Random Forest')

db = pd.read_csv("bhutan_landslide_data.csv")
db.drop(['TWI','STI','Slope','Slope length', 'Total curvature','Type'], axis=1, inplace=True)

X = db.drop(['Code','FID'], axis=1)
y = db['Code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42, stratify=y_train) #stratify is used for keepting the percentage ratio same while taking sampled from testing

rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

val_accuracy = []
for model in [rf]:
    val_accuracy.append(accuracy_score(y_val, model.predict(X_val)))

RFCacc = accuracy_score(y_val, rf.predict(X_val))

metrics = {
'RFC Acc': RFCacc
}

with open("metrics.json", 'w') as outfile:
        json.dump(metrics, outfile)

wandb.log(metrics)
pickle.dump(rf,open('LandslideRF.pkl','wb'))

# Use W&B's visualization tools to explore the results
wandb.finish()

