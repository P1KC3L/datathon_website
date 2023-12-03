from flask import Flask, render_template
import webbrowser

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_empleado')
def add_empleado():
    # Abre la página add.html en el navegador
    webbrowser.open("add.html")

@app.route('/buscar_empleado')
def buscar_empleado():
    # Abre la página search.html en el navegador
    webbrowser.open("search.html")

if __name__ == '__main__':
    app.run(debug=True)
