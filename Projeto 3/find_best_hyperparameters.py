import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
def load_db(file_path):
    return pd.read_csv(file_path)

def greedy_search_MultinomialNB(data, alpha_values):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['text'])  
    y = data['class']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []

    # Greedy search para o melhor valor de alpha
    best_alpha = None
    best_accuracy = 0

    for alpha in alpha_values:

        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        
        # Validar no conjunto de teste
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        
        # Armazenar resultados
        results.append({
            "alpha": alpha,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro
        })

    results_df = pd.DataFrame(results)
    output_file = "./find_best_hyperparameters_MultinomialNB.csv"
    results_df.to_csv(output_file, index=False)
    return results_df

    
def greedy_search_LogisticRegression(data, param_grid):
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['text'])  
    y = data['class']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # DataFrame para armazenar os resultados
    results = []

    # Greedy search para o melhor valor de C
    for c_value in param_grid:
        
        model = LogisticRegression(C=c_value, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Validar no conjunto de teste
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        
        # Armazenar resultados
        results.append({
            'C': c_value,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro
        })

    results_df = pd.DataFrame(results)
    output_file = "./find_best_hyperparameters_LogisticRegression.csv"
    results_df.to_csv(output_file, index=False)
    return results_df

def greedy_search_LinearSVC(data, param_grid):
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['text'])  
    y = data['class']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # DataFrame para armazenar os resultados
    results = []

    # Greedy search para o melhor valor de C
    for c_value in param_grid:
        
        model = LinearSVC(C=c_value, random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
       # Validar no conjunto de teste
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        
        # Armazenar resultados
        results.append({
            'C': c_value,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro
        })

    results_df = pd.DataFrame(results)
    output_file = "./find_best_hyperparameters_LLinearSVC.csv"
    results_df.to_csv(output_file, index=False)
    return results_df
