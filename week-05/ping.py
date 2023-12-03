from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

# Clicking on these won't work
# * Running on http://127.0.0.1:9696
# * Running on http://192.168.0.10:9696

# type http://localhost:9696/ping in the address bar

