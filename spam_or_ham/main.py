import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Load the model and vectorizer
bnb_loaded = joblib.load('bernoulli_nb_model.pkl')
tfidf_loaded = joblib.load('tfidf_vectorizer.pkl')

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess and transform the input message
def transform_Msg(Msg):
    Msg = Msg.lower()
    Msg = nltk.word_tokenize(Msg)
    y = [i for i in Msg if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Take input from the user
new_data = str(input("Enter a message: "))

# Preprocess and transform the input message
new_data_transformed = tfidf_loaded.transform([transform_Msg(new_data)]).toarray()

# Predict using the loaded model
prediction = bnb_loaded.predict(new_data_transformed)

# Output the result
if prediction == 0:
    print("Ham")
else:
    print("Spam")