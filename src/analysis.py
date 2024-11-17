import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils import tsne_plot, plot_feature_importance


files_and_labels = [
    ("../data/csv/Normal.csv", "normal"),
    # ("Talking.csv", "talking"),
    ("../data/csv/Yawning_cleaned.csv", "yawning"),
]

combined_csv = pd.concat(
    [pd.read_csv(file).assign(label=label) for file, label in files_and_labels],
    ignore_index=True
)

# X = combined_csv['jawOpen'].values.reshape(-1, 1)
X = combined_csv.iloc[:, 1:-1]
y = combined_csv['label']

tsne_plot(X, y, 'All features - TSNE')

# Feature importance
model = RandomForestClassifier(random_state=0)
model.fit(X, y)
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]
sorted_features = feature_names[indices]
sorted_importances = importances[indices]

plot_feature_importance(sorted_features, sorted_importances, n=10)




SELECTED_COLUMNS = ['jawOpen',
 'cheekPuff',
 'mouthLowerDownLeft',
 'jawRight',
 'mouthLowerDownRight',
 'mouthShrugLower',
 'mouthFunnel',
 'jawForward',
 'mouthClose',
 'mouthStretchLeft']

X_1 = X[SELECTED_COLUMNS]

tsne_plot(X_1, y, '10 most important features - TSNE')
