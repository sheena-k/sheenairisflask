from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# load the model
file = pickle.load(open("savemodel.pickle","rb"))

@app.route("/")
def home():
    result =""
    return render_template("index.html",**locals())

@app.route("/predict", methods=["POST","GET"])
def predict():
    Sepal_Length = float(request.form["sepal_length"])
    Sepal_Width = float(request.form["sepal_width"])
    Petal_Length = float(request.form["petal_length"])
    Petal_Width = float(request.form["petal_width"])
    result = file.predict([[Sepal_Length,Sepal_Width,Petal_Length,Petal_Width]])[0]
    return render_template("result.html", **locals())




if __name__ == "__main__":
    app.run(debug=True)