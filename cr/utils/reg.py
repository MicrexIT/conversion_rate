from sklearn.model_selection import train_test_split


def split_data(data, target, stratify=False, ts=0.3, rs=42):
    X = data.drop(columns=[target])
    y = data[target]
    s = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs, test_size=ts, stratify=s)
    return [X_train, X_test, y_train, y_test]
