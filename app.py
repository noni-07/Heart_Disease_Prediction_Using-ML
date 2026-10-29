from flask import Flask, request, render_template
import pickle
import numpy as np 

app=Flask(__name__)

with open("suv.pkl", 'rb') as model_file:
	ml_model=pickle.load(model_file)

@app.route("/")
def index():
	return render_template('index.html')


@app.route("/process", methods=["POST"])
def process():

    features = [
        float(request.form["Age"]),
        float(request.form["Sex"]),
        float(request.form["Chestpaintype"]),
        float(request.form["BP"]),
        float(request.form["Cholesterol"]),
        float(request.form["FBover120"]),
        float(request.form["EKGresults"]),
        float(request.form["MaxHR"]),
        float(request.form["Exerciseangina"]),
        float(request.form["STdepression"]),
        float(request.form["SlopeofST"]),
        float(request.form["Numberofvesselsfluro"]),
        float(request.form["Thallium"]),
    ] 
                        

    input_data = np.array(features).reshape(1, -1)

    result = ml_model.predict(input_data)[0]

    diagnosis = "Heart Disease" if result == 1 else "No Heart Diseaseiabetes"

    return render_template("output.html", prediction=diagnosis)











if __name__=='__main__':
	app.run(debug=True)