
import json
from flask import Flask, request, jsonify, make_response
from utils import createRandomFileID
import os
import subprocess
import time
DATA_FOLDER = 'incoming_data'
if not os.path.isdir(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024 #200mb file limit


@app.route('/test')
# @cross_origin()
def home():
    status=200
    response = make_response(
                jsonify(
                    {"message": 'Hello!'}
                ),
                status,
            )
    return response

@app.route('/caption/final/<model>/<output_type>', methods=['POST'])
def get_captions(model, output_type):
    try:
        print(model, output_type)
        new_file_id = createRandomFileID()
        while os.path.isdir(new_file_id):
            new_file_id = createRandomFileID()
        new_path = os.path.join(DATA_FOLDER, new_file_id)
        if not os.path.isdir(DATA_FOLDER):
            os.mkdir(DATA_FOLDER)
        os.mkdir(new_path)
        assert os.path.isdir(new_path)

        # print(request.data)
        # print(request.form)

        # curl --form "file=@test.mp4" http://127.0.0.1:5000/caption/final/json/
        audio = request.files.get('file')
        # d = request.data
        # data = request.get_json()
        
        #save incoming mp4 to audio.mp4
        audio_path = os.path.join(new_path, 'audio.mp4')
        metadata = json.dumps({'time_received': f"{time.time()}", 'output_type': f"{output_type}"})
        with open(os.path.join(new_path, 'metadata.txt'), 'w') as f:
            f.write(metadata)

        # with open(audio_path, 'w'):
        audio.save(audio_path)

        audio.close()
        subprocess.Popen(["python","main.py","--demo", f"{audio_path}", "-k", "3", "-m", f"{model}", "-p", f"{model}:" ])
        # os.system(f"python main.py --demo {audio_path} -k 5")      

        response = make_response(
                    jsonify(
                        {"file_id": f'{new_file_id}'}
                    ),
                    200,
                )
        return response

    except Exception as e:
        status = 400
        print(e)
        response = make_response(
                    jsonify(
                        {"message": f'{e}'}
                    ),
                    status,
                )
        return response

@app.route('/getJobStatus/<output_type>/<file_id>', methods=['GET'])
def get_job_status(output_type, file_id):
    #   curl -X GET http://127.0.0.1:5000/getJobStatus/xml/5t3hvydy-bibo-3mb4-jm26-teqrlzhnjv41
    try:
        with open(os.path.join(DATA_FOLDER, file_id, 'metadata.txt'), 'r') as f:
            metadata = json.load(f)
        check_path = os.path.join(DATA_FOLDER, file_id, f'audio.{output_type}')
        if os.path.exists(check_path):
            status = 'Done'
        else:
            status = 'Not Done'
            if (time.time() - int(metadata['time_received'])) > 1800:   #if more than 30min, probably error happened.
                status = "Timeout" #Something wrong happened. send in the audio again.


        response = make_response(
                    jsonify(
                        {"status": status}
                    ),
                    200,
                )
        return response

    except Exception as e:
        status = 400
        print(e)
        response = make_response(
                    jsonify(
                        {"message": 'Error'}
                    ),
                    status,
                )
        return response

@app.route('/returnFile/<output_type>/<file_id>', methods=['GET'])
def return_file(output_type, file_id):
    # curl -X GET http://127.0.0.1:5000/returnFile/xml/5t3hvydy-bibo-3mb4-jm26-teqrlzhnjv41
    try:
        check_path = os.path.join(DATA_FOLDER, file_id, f'audio.{output_type}')
        with open(check_path,'r') as f:
            return_data = f.read()
        response = make_response(return_data, 200)

        return response

    except Exception as e:
        status = 400
        print(e)
        response = make_response(
                    jsonify(
                        {"message": 'Not ready!'}
                    ),
                    status,
                )
        return response

if __name__ == "__main__":
    app.run('0.0.0.0',port=5050, debug=True)
    # serve(app, port=8080)

