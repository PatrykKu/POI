import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

df = pd.read_csv('texture_features.csv')

#features (X) and labels (y)
X = df.drop('category', axis=1).values
y = df['category'].values

#integer encoding for labels
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)

#one-hot
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_int.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.3, random_state=42
)

#model
model = Sequential()
model.add(Dense(10, activation='sigmoid', input_dim=72))  # 72 features
model.add(Dense(3, activation='softmax'))  # 3 classes
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

#training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=10,
    shuffle=True,
    verbose=1
)

y_pred = model.predict(X_test)
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)

#confusion matrix
cm = confusion_matrix(y_test_int, y_pred_int)
print("Confusion Matrix:")
print(cm)