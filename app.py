from flask import Flask, request, jsonify
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
print("🔄 Loading model and vectorizer...")
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
print("✅ Model and vectorizer loaded!")

# Setup Flask app
app = Flask(__name__)

# Preprocessing function
ps = PorterStemmer()
def preprocess(text):
    print("🔧 Preprocessing text...")
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    processed = " ".join(text)
    print(f"✅ Processed text: {processed[:100]}...")  # Show first 100 chars
    return processed

# Route
@app.route('/predict', methods=['POST'])
def predict():
    print("📥 /predict endpoint hit!")

    try:
        data = request.get_json()
        print(f"📝 Received data: {data}")

        news_text = data.get("text")
        if not news_text:
            print("⚠️ No text provided in request.")
            return jsonify({"error": "No text provided"}), 400

        processed_text = preprocess(news_text)
        vectorized_input = vectorizer.transform([processed_text]).toarray()
        prediction = model.predict(vectorized_input)[0]
        confidence = model.predict_proba(vectorized_input)[0].max()

        result = "REAL" if prediction == 1 else "FAKE"
        print(f"📊 Prediction: {result}, Confidence: {confidence:.4f}")

        return jsonify({
            "prediction": result,
            "confidence": float(round(confidence, 4))
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
