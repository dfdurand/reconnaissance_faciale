from flask import Flask
from app  import views


app = Flask(__name__) #webserver gateway interphase

app.add_url_rule(rule='/', endpoint='home', view_func=views.index)

app.add_url_rule(rule='/app', endpoint='app', view_func=views.app)

app.add_url_rule(rule='/app/gender/', endpoint='gender',view_func=views.genderapp,  methods= ['GET', 'POST'])

app.add_url_rule(rule='/app/names/', endpoint='names',view_func=views.names,  methods= ['GET', 'POST'])

app.add_url_rule(rule='/app/test/', endpoint='test',view_func=views.test,  methods= ['GET', 'POST'])

app.add_url_rule(rule='/app/receive/', endpoint='receive',view_func=views.receive,  methods= ['GET', 'POST'])

# @app.route('/')
# def index():
#     return "welcome to my life on my website"

#



if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)