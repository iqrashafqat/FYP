from flask import Flask, render_template, Response,jsonify,request,Response
import cv2
import face_recognition
import numpy as np
import os
import sys
from flask_cors import CORS, cross_origin

dirname = os.path.dirname(os.path.abspath(__file__)) 
# dirname_list = dirname.split("/")[:-1]
# dirname = "/".join(dirname_list)
# print("DIRNAME: ",dirname)
images_dir_path = dirname + "/" + "images"
print("IMage DIr Name: ",images_dir_path)


app=Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=[
    "Content-Type", "Authorization","Access-Control-Allow-Origin", "Access-Control-Allow-Credentials","application/json"],
    supports_credentials=True)

camera = cv2.VideoCapture(0) 

# Initialize some variables
known_face_encodings = []
known_face_names = []

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
encodings_dict_index = dict()

print("known face names: ",known_face_names)
print("known face encoding: ",known_face_encodings)

print("**************************************************************************")

def get_known_face_encodings():
    print("INside Encoding Function")

    folder_names = os.listdir(images_dir_path)
    
    for index in range(len(folder_names)):
        f_name = folder_names[index]
        f_name_wth_ext = f_name.split(".")[0]

        f_path = images_dir_path + "/" + f_name
        print("-------------------------------------------------------------------")
        print("File path: ",f_path)
        print("-------------------------------------------------------------------")
      
        #Load a sample picture and learn how to recognize it.
        known_image = face_recognition.load_image_file(f_path) 
        known_face_encoding = face_recognition.face_encodings(known_image)[0]

        print(known_face_encoding)

        known_face_names.append(f_name_wth_ext)
        known_face_encodings.append(known_face_encoding)

        encodings_dict_index[f_name_wth_ext] = index

    print("Done")
        



def gen_frames():  
    get_known_face_encodings()
    print("known face names: ",known_face_names)
    print("known face encoding: ",known_face_encodings)
    print("Dictionary: ",encodings_dict_index)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            
            

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/') 
def index():
    return render_template('index2.html') 

@app.route('/camera_view') 
def camera_view():
    return render_template('index.html') 

@app.route('/add_image',methods=['POST'])
def add_image():
    try:
        Image=request.files["image"].read()
        name=request.form['person_name']
        print(name)
        print(Image)

        npimg = np.fromstring(Image, np.uint8)

        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)  

        f_name_wth_ext = name    
        name=name+".jpg" 

        f_path = os.path.join(images_dir_path , name)
        cv2.imwrite(f_path, img) 
    
        #Load a sample picture and learn how to recognize it.
        known_image = face_recognition.load_image_file(f_path) 
        known_face_encoding = face_recognition.face_encodings(known_image)[0]

        index = len(known_face_names)
        known_face_names.append(f_name_wth_ext)
        known_face_encodings.append(known_face_encoding)
        
        
        encodings_dict_index[f_name_wth_ext] = index
        print("Updated Dictionary: ",encodings_dict_index)

        dic = {"status":200,"msg":"ok"}

        return jsonify(dic)

    except Exception as e:
        print("Exception: ",str(e))

        dic = {"status":403,"msg":"Face Not Found","error":str(e)}

        return jsonify(dic)


@app.route('/delete_image',methods=['POST'])
def delete_image():
    try:
        name=request.form['person_name']
        f_name=name+".jpg" 
        f_path = os.path.join(images_dir_path , f_name)

        print("Length: ",len(known_face_names))

        index = encodings_dict_index[name]
        print("Index: ",index)

        del known_face_names[index]
        del known_face_encodings[index]
        del encodings_dict_index[name]

        os.remove(f_path)

        print("Updated Dictionary: ",encodings_dict_index)
        print("Length now: ",len(known_face_names))

        dic = {"status":200,"msg":"ok"}

        return jsonify(dic) 

    except Exception as e:

        print("Exception: ",str(e))

        dic = {"status":403,"msg":"Face Not Found","error":str(e)}

        return jsonify(dic)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True,debug=False) 