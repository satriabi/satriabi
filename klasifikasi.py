import numpy as np

def count_vectorizer(emails, vocabulary=None):
    if vocabulary is None:
        # Proses teks menggunakan CountVectorizer
        unique_words = set(word for email in emails for word in email.split())
        vocabulary = {word: index for index, word in enumerate(unique_words)}

    # Membuat vektor untuk setiap email
    vectorized_emails = np.zeros((len(emails), len(vocabulary)))

    for i, email in enumerate(emails):
        for word in email.split():
            if word in vocabulary:
                vectorized_emails[i, vocabulary[word]] += 1

    return vectorized_emails, vocabulary

def train_naive_bayes(X_train, y_train):
    # Hitung probabilitas prior
    total_samples = len(y_train)
    spam_prior = np.sum(y_train) / total_samples
    normal_prior = 1 - spam_prior

    # Hitung likelihood
    spam_likelihood = np.sum(X_train[y_train == 1], axis=0) + 1
    normal_likelihood = np.sum(X_train[y_train == 0], axis=0) + 1

    # Normalisasi likelihood
    spam_likelihood /= np.sum(spam_likelihood)
    normal_likelihood /= np.sum(normal_likelihood)

    return spam_prior, normal_prior, spam_likelihood, normal_likelihood

def predict_naive_bayes(email_vector, spam_prior, normal_prior, spam_likelihood, normal_likelihood):
    # Hitung posterior
    spam_posterior = np.log(spam_prior) + np.sum(np.log(spam_likelihood) * email_vector)
    normal_posterior = np.log(normal_prior) + np.sum(np.log(normal_likelihood) * email_vector)

    # Prediksi label
    if spam_posterior > normal_posterior:
        return 1  # Spam
    else:
        return 0  # Normal

def predict_email_category(new_email, vocabulary, spam_prior, normal_prior, spam_likelihood, normal_likelihood):
    new_email_vectorized, _ = count_vectorizer(np.array([new_email]), vocabulary)
    prediction = predict_naive_bayes(new_email_vectorized[0], spam_prior, normal_prior, spam_likelihood, normal_likelihood)
    
    if prediction == 0:
        return 'Email ini tidak masuk kategori spam.'
    else:
        return 'Email ini masuk kategori spam.'

# Data contoh dengan variasi yang lebih besar
data = {
    'email': ['Halo, ini adalah email normal.',
              'Menangkan hadiah besar sekarang!',
              'Tolong konfirmasi jadwal pertemuan.',
              'Promo spesial untuk Anda!',
              'Ini adalah email biasa.',
              'Penting: Perubahan jadwal pertemuan pekan ini.',
              'Promo eksklusif hanya untuk pelanggan setia!'],
    'label': [0, 1, 0, 1, 0, 0, 1]  # 0: Normal, 1: Spam
}

# Ubah data menjadi array NumPy
emails = np.array(data['email'])
labels = np.array(data['label'])

# Proses teks menggunakan count_vectorizer
X_train, vocabulary = count_vectorizer(emails)

# Latih model Naive Bayes
spam_prior, normal_prior, spam_likelihood, normal_likelihood = train_naive_bayes(X_train, labels)

# Contoh penggunaan model untuk prediksi email baru
new_email = 'Halo, ini adalah email dari satriairawan203 dengan informasi penting.'
result = predict_email_category(new_email, vocabulary, spam_prior, normal_prior, spam_likelihood, normal_likelihood)
print(result)