import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

def create_model(data):
    x = data.drop(['y'], axis=1)
    y = data['y']
    
    # scale data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    # split data
    
    x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # train
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    # test
    y_pred = model.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred) )
    print("Classification report: \n", classification_report(y_test, y_pred ))
    
    return model, scaler



      
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 0'], axis=1)
    data['y'] = data['y'].map({'M':1, 'B':0})
    return data


def main():
    data = get_clean_data()
    model, scaler = create_model(data)
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    # train(model)
    
    # evaluate(model)
    

    
    
if __name__ == '__main__':
    main()