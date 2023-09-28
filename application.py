from flask import Flask, request, render_template
from src.Turbo_Engine_Predict_Maintenance.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoints():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
                 engine_number =        float(request.form.get("engine_number")),
                 time_cycles =          float(request.form.get("time_cycles")),
                 sensor_measurement2 =  float(request.form.get("sensor_measurement2")),
                 sensor_measurement3 =  float(request.form.get("sensor_measurement3")),
                 sensor_measurement4 =  float(request.form.get("sensor_measurement4")),
                 sensor_measurement7 =  float(request.form.get("sensor_measurement7")),
                 sensor_measurement8 =  float(request.form.get("sensor_measurement8")),
                 sensor_measurement9 =  float(request.form.get("sensor_measurement9")), 
                 sensor_measurement11 = float(request.form.get("sensor_measurement11")),
                 sensor_measurement12 = float(request.form.get("sensor_measurement12")),
                 sensor_measurement13 = float(request.form.get("sensor_measurement13")),
                 #sensor_measurement15 = float(request.form.get("sensor_measurement15")),
                 #sensor_measurement17 = float(request.form.get("sensor_measurement17")),
                 #sensor_measurement20 = float(request.form.get("sensor_measurement20")),
                 #sensor_measurement21 = float(request.form.get("sensor_measurement21"))
            )
            

        final_new_data = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results=round(pred[0])

        return render_template('results.html',final_result=results)


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True,port=5001)