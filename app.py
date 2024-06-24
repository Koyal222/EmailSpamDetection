from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
cv = pickle.load(open("models/cv.pkl", 'rb'))
clf = pickle.load(open("models/clf.pkl", 'rb'))

@app.route('/', methods=["GET", "POST"])
def home():
    text=""
    if request.method == "POST":
        text=request.form.get("email-content") 
    return render_template('index.html', text=text)


@app.route('/predict', methods=['POST'])
def predict():
        email = request.form.get('email-content')
        # predict email
        # print(email)
        X = cv.transform([email])
        prediction = clf.predict(X)
        prediction = 1 if prediction == 1 else -1
        return render_template('index.html', response=prediction, text=email)

if __name__ == '__main__':
    app.run(debug=True)
