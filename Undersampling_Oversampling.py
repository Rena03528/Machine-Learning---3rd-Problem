def undersampling(X_train,y_train):
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    rus = RandomUnderSampler(random_state=42)
    X_train_us, y_train_us = rus.fit_resample(X_train_flat, y_train)

    X_train_us = X_train_us.reshape(X_train_us.shape[0], 48, 48, 1)

    y_train_us= to_categorical(y_train_us, 2)

    return X_train_us, y_train_us

def oversampling(X_train, y_train):
    # Flatten the input data for SMOTE
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  

    # Initialize SMOTE and apply it to the training data
    smote = SMOTE(random_state=42)
    X_train_os, y_train_os = smote.fit_resample(X_train_flat, y_train)

    # Reshape the oversampled features back to their original dimensions
    X_train_os = X_train_os.reshape(X_train_os.shape[0], 48, 48, 1) 

    # One-hot encode the labels
    y_train_os = to_categorical(y_train_os, num_classes=2) 

    # Check shapes for consistency
    print(f"Original X_train shape: {X_train.shape}")
    print(f"Original y_train shape: {y_train.shape}")
    print(f"Oversampled X_train_os shape: {X_train_os.shape}")
    print(f"Oversampled y_train_os shape: {y_train_os.shape}")

    # Ensure that the number of samples is the same
    if X_train_os.shape[0] != y_train_os.shape[0]:
        raise ValueError(f"Inconsistent sample sizes: {X_train_os.shape[0]} in X_train_os and {y_train_os.shape[0]} in y_train_os")

    return X_train_os, y_train_os
