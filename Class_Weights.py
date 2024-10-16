def weigthed_classes(y_train,y_val):

    y_train = np.asarray(y_train).ravel()  # Flatten if necessary
    y_val = np.asarray(y_val).ravel()  # Flatten if necessary

    class_labels = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=class_labels, y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_labels))}

    print("Class weights:", class_weight_dict)

    y_train_categorical = to_categorical(y_train, num_classes=len(class_labels))
    y_val_categorical = to_categorical(y_val, num_classes=len(class_labels))

    return y_train_categorical, y_val_categorical
