def tests():
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['task1'], test_size=0.2, random_state=42)
    # Define the vocabulary
    vocab = set()
    for sentence in X_train:
        words = sentence.split()
        for word in words:
            vocab.add(word)

    # Create a dictionary to store the count of each word in each class
    class_word_counts = {}
    for c in np.unique(y_train):
        class_word_counts[c] = {}
        for word in vocab:
            class_word_counts[c][word] = 0

    # Count the number of occurrences of each word in each class
    for i in range(len(X_train)):
        words = X_train.iloc[i].split()
        c = y_train.iloc[i]
        for word in words:
            class_word_counts[c][word] += 1

    # Prior probability of a class
    class_priors = {}
    for c in np.unique(y_train):
        class_priors[c] = len(y_train[y_train == c]) / len(y_train)

    # Compute the total count of words in each class
    class_word_totals = {}
    for c in np.unique(y_train):
        class_word_totals[c] = sum(class_word_counts[c].values())

    # Define a function to predict the class of a new text sample
    def predict(text):
        words = text.split()
        probs = {}
        for c in np.unique(y_train):
            # log_prob = np.log(class_priors[c])
            log_prob = 1;
            for word in words:
              count = 1
              if word in vocab:
                  count += class_word_counts[c][word] + 1 # Laplace smoothing
              log_prob += np.log(count / (class_word_totals[c] + len(vocab)))
            probs[c] = log_prob
        return max(probs, key=probs.get)

    # Evaluate the performance of the classifier on the testing data
    correct = 0
    for i in range(len(X_test)):
        pred = predict(X_test.iloc[i])
        if pred == y_test.iloc[i]:
            correct += 1
    accuracy = correct / len(X_test)
    print("Accuracy:", accuracy)
    return accuracy