from flask import Flask
from flask import render_template, request, jsonify
from eval import get_model_api

app = Flask(__name__)

#load the model
model_api = get_model_api()

@app.route('/api', methods=['POST'])
def api():
    """API function
    All model-specific logic to be defined in the get_model_api()
    function
    """
    input_data = request.json
    app.logger.info("api_input: " + str(input_data))
    output_data = model_api(input_data)
    app.logger.info("api_output: " + str(output_data))
    response = jsonify(output_data)
    return response


@app.route('/')
def index():
    return render_template('full_client.html')

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500



app.run(host='0.0.0.0', debug=True)