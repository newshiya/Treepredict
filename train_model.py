import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("formupredict_sample_dataset.csv")
features = df[['Binder', 'Disintegrant', 'Ratio_Binder_Dis', 'Compression_Force', 'pH', 'Hardness']]
y_dis = df['Disintegration_Time_sec']
y_diss = df[['%_Release_2h', '%_Release_4h', '%_Release_6h']]

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), ['Binder', 'Disintegrant', 'Compression_Force'])
], remainder='passthrough')

# Train disintegration model
dis_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100))
])
dis_model.fit(features, y_dis)
joblib.dump(dis_model, 'disintegration_model.pkl')

# Train dissolution model
X_transformed = preprocessor.fit_transform(features)
nn_model = Sequential([
    Dense(64, activation='relu', input_dim=X_transformed.shape[1]),
    Dense(64, activation='relu'),
    Dense(3)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_transformed, y_diss, epochs=100, verbose=0)
nn_model.save("dissolution_model.h5")