from flask import Flask, request, render_template
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import os
from werkzeug.utils import secure_filename
from collections import Counter
from transformers import pipeline
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the summarization model from HuggingFace
summarizer = pipeline("summarization")

def summarize_text(text, n=3, use_ai=False):
    if use_ai:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    else:
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text)
        word_frequencies = Counter(word.lower() for word in words if word.lower() not in stop_words and word.isalnum())
        max_frequency = max(word_frequencies.values(), default=1)
        for word in word_frequencies:
            word_frequencies[word] /= max_frequency

        # Tokenize sentences
        sentence_list = sent_tokenize(text)
        sentence_scores = {}
        for sentence in sentence_list:
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    if len(sentence.split(' ')) < 30:
                        if sentence not in sentence_scores:
                            sentence_scores[sentence] = word_frequencies[word]
                        else:
                            sentence_scores[sentence] += word_frequencies[word]

        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:n]
        summary = ' '.join(summary_sentences)
        return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    if request.method == 'POST':
        use_ai = 'use_ai' in request.form
        if 'article_file' in request.files:
            file = request.files['article_file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                with open(filepath, 'r') as f:
                    text = f.read()
                summary = summarize_text(text, use_ai=use_ai)
        elif 'article' in request.form:
            text = request.form['article']
            summary = summarize_text(text, use_ai=use_ai)
    return render_template('index.html', summary=summary)

if __name__ == "__main__":
    app.run(debug=True)