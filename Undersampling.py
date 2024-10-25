def undersampling(X_train,y_train):
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    rus = RandomUnderSampler(random_state=42)
    X_train_us, y_train_us = rus.fit_resample(X_train_flat, y_train)

    X_train_us = X_train_us.reshape(X_train_us.shape[0], 48, 48, 1)

    y_train_us= to_categorical(y_train_us, 2)

    return X_train_us, y_train_us
