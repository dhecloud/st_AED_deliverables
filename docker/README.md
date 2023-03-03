# Docker Instructions

This details how to set up the docker image and run the repository with 3 API endpoints. This docker image is meant for easy deployment for the models. If you want to modify the code/config/checkpoints, you should work directly on the git repository. 

## Setting up

### Docker and docker-compose

```
# Install docker engine
curl -fsSL https://get.docker.com -o get-docker.sh
sh ./get-docker.sh

# Install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add docker to sudoers so don’t need to run sudo every time using docker
sudo groupadd docker
sudo usermod -aG docker $USER
```

### Install and start AED application
1. `git clone https://github.com/dhecloud/st_AED_deliverables.git`
2. `cd docker`
3. `docker build -t st-aed:aug2022 .`

Alternatively, you could simply copy `Dockerfile` from this folder to your local machine and run `docker build .`

After the image is done building, get the `image_id` (st-aed:aug2022) and run `docker run -t -d -p <your_port>:5050 <image_id>`, or `docker-compose -f docker-compose.yaml up -d`

(Option -d: detachable)

Check [Functions and usage](#Functions-and-usage) for further details.


## Model available

There is only 1 available model in this commit, namely A2. For previous models, please refer to previous commits.

1. A2 - mobilenetv2 trained on our final processed dataset


`A2` predicts these classes - breaking, crowd_scream, crying_sobbing, explosion, gunshot_gunfire, motor_vehicle_road, siren, speech, silence.  



## Functions and usage

Note that the Dockerfile includes an entrypoint command. If u want to modify the code/config/checkpoints, you should use the git repo and not docker. However, if you still want to use docker for some reason, then comment out the `ENTRYPOINT` command on the last line of the `Dockerfile`. Then you can run the container interactively with the `-i` flag.

The dockerfile sets up the working environment along with the apis. The model checkpoints are included. An image should use around 4GB of memory space.

There are 3 endpoints:

1. `<ip_address>:<your_port>/caption/final/A2/<output_type>` - use this for incoming new audio clips
2. `<ip_address>:<your_port>/getJobStatus/<output_type>/<file_id>` - use this to get job status/completion
3. `<ip_address>:<your_port>/returnFile/<output_type>/<file_id>` - use this to get the file containing the predicted classes

Currently, `output_type` can be `xml`, `srt`, or `json`. You can use any REST api client to send requests to the endpoints. 

### caption\/final\/\<model\>/\<output_type\>


This endpoint takes in a file with a POST request and creates its corresponding captions in `srt`, `xml` and `json` format. If the audio file is received successfully, a `file_id` will be returned. `wav`, `mp3` and `mp4` formats are supported and all other formats will be ignored.

For instance, if `your_port` is `5050`, `model` is `A2`, i can send an wav file to the endpoint via `curl --form "file=@test_1min.wav" http://127.0.0.1:5050/caption/final/A2/json` and it will return

```
{
  "file_id": "9jdepho3-e1c0-oi3l-nm6r-vt2bjhyswx4i"
}
```

Notes:
   - `A2` if you want to submit job to `A2` model.
   - json in the path is just the response format (the file id return as JSON object)


Note that the successful response of `file_id` does not indicate that the `srt`, `xml` and `json` files have been created. Once the audio file has been recieved, the audio clip will be processed. How long it takes depends on the compute power available and the length of the audio clip.

### getJobStatus\/\<output_type\>\/\<file_id\>

Use a GET request. Using the `file_id` from `caption/final/<output_type>`, you can use this endpoint to check the status of the audio clip. There are three possible message status:
 
1. `Done` - The audio clip has finished processing
2. `Not Done` - The audio clip has not finished processing
3. `Timeout` - Something wrong happened. Send in the audio again. This happpens when there is an error during the prediction process. Timeout will only be returned if there are no outputs 30min after the initial receiving of the audio clip

For instance, using the same `file_id`, i can do `curl -X GET http://127.0.0.1:5050/getJobStatus/xml/9jdepho3-e1c0-oi3l-nm6r-vt2bjhyswx4i` and it will return

```
{
  "status": "Done"
}
```

### returnFile\/\<output_type\>\/\<file_id\>

Use a GET request to retrieve the file containing the predicted classes. 

For instance, if i want to get predictions in `xml` format, i can do `curl -X GET http://127.0.0.1:5050/returnFile/xml/9jdepho3-e1c0-oi3l-nm6r-vt2bjhyswx4i`


<details><summary>Click to see xml response</summary>
<p>
```
&lt;?xml version="1.0" ?&gt;
&lt;AudioDoc name="9jdepho3-e1c0-oi3l-nm6r-vt2bjhyswx4i"&gt;
    &lt;SoundCaptionList&gt;
        &lt;SoundSegment stime="0.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="1.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="2.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="3.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="4.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="5.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: motor_vehicle_road 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="6.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="7.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="8.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="9.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="10.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="11.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="12.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="13.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="14.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="15.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="16.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="17.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="18.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="19.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="20.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="21.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: crying_sobbing &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="22.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="23.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="24.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="25.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="26.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="27.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="28.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="29.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="30.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="31.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="32.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="33.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="34.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="35.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="36.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="37.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="38.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="39.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="40.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="41.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="42.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="43.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="44.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="45.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="46.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="47.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="48.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="49.0" dur="1.00"&gt;M1: 0: others 1: motor_vehicle_road 2: chatter 3: emergency_vehicle 4: siren &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="50.0" dur="1.00"&gt;M1: 0: others 1: motor_vehicle_road 2: siren 3: emergency_vehicle 4: screaming &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="51.0" dur="1.00"&gt;M1: 0: others 1: chatter 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="52.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="53.0" dur="1.00"&gt;M1: 0: chatter 1: screaming 2: others 3: emergency_vehicle 4: breaking &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="54.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="55.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: emergency_vehicle 4: motor_vehicle_road &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="56.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="57.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="58.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: motor_vehicle_road 3: screaming 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="59.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: motor_vehicle_road 3: screaming 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="60.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: motor_vehicle_road 3: screaming 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="61.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="62.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="63.0" dur="1.00"&gt;M1: 0: chatter 1: others 2: screaming 3: motor_vehicle_road 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="64.0" dur="1.00"&gt;M1: 0: others 1: motor_vehicle_road 2: chatter 3: screaming 4: emergency_vehicle &lt;/SoundSegment&gt;
        &lt;SoundSegment stime="65.0" dur="1.00"&gt;M1: 0: others 1: motor_vehicle_road 2: siren 3: screaming 4: chatter &lt;/SoundSegment&gt;
    &lt;/SoundCaptionList&gt;
&lt;/AudioDoc&gt;
```
</p>
</details>  


Note that you should pipe the response of this endpoint to a file.


## FAQ

#### Is there a limitation on length of audio clip?
No, there is no limitation. However, it does depends on your machine compute power. A 1 min clip should take around 10 to 20 seconds to fully process. It is recommended to break extremely long audio clips into smaller parts.

#### Can i change the model or use a different model checkpoint?
Yes, but not in this docker image. This docker image is meant for easy deployment. Modify the repository directly for any changes. If you are familiar with docker, then you can work within the container. 

#### How do i interpret the results?
The model predicts for every 3 second window and outputs a caption every 1 second. Therefore, there is a prediction every second. The top 3 most probable classes will be returned. `A2: 0: silence 1: motor_vehicle_road 2: siren` is an example of a prediction. `A2` is the model name, `0:` indicates that `silence` is the most probable class, followed by `1:` which indicates `motor_vehicle_road` is the next probable class, and so on.

#### Error catching
Feel free to contact me if there are any errors for which you cannot solve.



