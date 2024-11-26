import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file_path = r'C:\Users\ACER\Desktop\Suhas\Internship\CodeSoft\IRIS.csv'
iris_data = pd.read_csv(file_path, encoding='latin1')
X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

def predict_iris_species():
    sepal_length = float(input("Enter Sepal Length (cm): "))
    sepal_width = float(input("Enter Sepal Width (cm): "))
    petal_length = float(input("Enter Petal Length (cm): "))
    petal_width = float(input("Enter Petal Width (cm): "))
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                              columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    prediction = model.predict(input_data)
    print(f"Predicted Iris species: {prediction[0]}")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

predict_iris_species()
