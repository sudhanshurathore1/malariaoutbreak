from flask import Flask,request, url_for, redirect, render_template
from sklearn.metrics import _dist_metrics
import pickle
import numpy as np
import requests


app = Flask(__name__)




@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict_class():
    print([(x) for x in request.form.values()])
    features = [(x) for x in request.form.values()] #maintain the input same as the data that u trained model
    city=features[0]
    rainfall=int(features[1])
    print(city)
    print(rainfall)
    api_key1 = 'ad62ecebb7931902c9fdbfefb78f3277'
    url = 'http://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metric'.format(city, api_key1)
    res = requests.get(url)
    data = res.json()

    print(f"Place : {data['name']}")
    latitude = data['coord']['lat']
    longitude = data['coord']['lon']
    print('latitude :', latitude)
    print('longitude :', longitude)
    # getting the main dict block
    main = data['main']
    wind = data['wind']
    # getting temperature
    temperature = main['temp']
    # getting the humidity
    humidity = main['humidity']
    tempmin = main['temp_min']
    tempmax = main['temp_max']
    # getting the pressure
    windspeed = wind['speed']
    pressure = main['pressure']
    # weather report
    report = data['weather']
    print(f"Temperature : {temperature}°C")
    print(f"Temperature Min : {tempmin}")
    print(f"Temperature Max : {tempmax}")
    print(f"Humidity : {humidity}")
    print(f"Pressure : {pressure}")
    print(f"Wind Speed : {windspeed}")
    print(f"Weather Report : {report[0]['description']}")

    inputt = []
    inputt.append(tempmax)
    inputt.append(tempmin)
    inputt.append(humidity)
    inputt.append(rainfall)

    final = [np.array(inputt)]
    print(final)
    model = pickle.load(open('model.pkl', 'rb'))
    sst = pickle.load(open('sst.pkl', 'rb'))
    prediction = model.predict_proba(sst.transform(final))
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    print(output)

    if output >= str(0.5):
        return render_template('index.html',
                               pred='There are more chances of Malaria Outbreak. \nProbability of Malaria Breeds occuring is {}.\n '.format(
                                   output),
                               inp="The Prediction is based on: Max Temp: %.2f,Min Temp: %.2f,Humidity: %.1f Units(Celcius,Percent) and Rainfall: %d Units" % (
                               inputt[0], inputt[1], inputt[2],inputt[3]))
    else:
        return render_template('index.html',
                               pred='Less chance of Malaria Outbreak\n Probability of Malaria Breeds occuring is {}\n '.format(
                                   output),
                               inp="The Prediction is based on: Max Temp: %.2f,Min Temp: %.2f,Humidity: %.1f Units(Celcius,Percent) and Rainfall: %d Units" % (
                               inputt[0], inputt[1], inputt[2],inputt[3]))
    #     sst = pickle.load(file)
    #
    # output= clf.predict(sst.transform([features]))
    # print(output)
    # if output[0]==0:
    #     return render_template("index.html",pred="The Person will not Purchase the SUV")
    #
    # else:
    #     return render_template("index.html", pred="The Person will Purchase the SUV")


if __name__=="__main__":
    app.run(debug=True) #would create a flask local server