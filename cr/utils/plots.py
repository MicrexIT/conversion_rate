from sklearn.model_selection import train_test_split


def split_data(data, target, test_size=0.3, random_state=42):
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    return [X_train, X_test, y_train, y_test]
