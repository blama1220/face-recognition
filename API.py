from flask import Flask, request, redirect, render_template, Response
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import cv2
import subprocess
import face_recognition as fr
import numpy as np
from time import sleep


crr = ""
now = datetime.now()
app = Flask(
    __name__,
    template_folder="template",
    static_folder="static",
)
file_name = "faces"
app.config["IMAGE_UPLOADS"] = os.path.abspath(file_name)
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG", "JPG", "JPEG"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024


def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # img = img[:,:,::-1]

    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(
                img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2
            )

            # Draw a label with a name below the face
            cv2.rectangle(
                img,
                (left - 20, bottom - 15),
                (right + 20, bottom + 20),
                (255, 0, 0),
                cv2.FILLED,
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                img, name, (left - 20, bottom + 15), font, 1.0, (255, 255, 255), 2
            )

    # Display the resulting image
    # while True:

    #     cv2.imshow('Video', img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         return face_names
    # pathres = 'D:/OpenCV/Scripts/Images'
    # cv2.imwrite(os.path.join(path , 'waka.jpg'), img)

    dt_string = now.strftime("%d%m%Y%H%M%S")
    imagename = "{}{}.jpg".format("test",dt_string)  # could be a uuid instead of datetime

    cv2.imwrite(imagename, img)
    os.system('move ' + imagename + ' static')
    global crr
    crr = imagename
    print(imagename)


@app.route("/")
def index():
    return render_template("index.html")


def allowed_image(filename):
    if not "." in filename:
        return False
    extension_file = filename.rsplit(".", 1)[1]
    if extension_file.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def allowed_image_filesize(filesize):
    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":

        if request.files:

            if not allowed_image_filesize(request.cookies.get("filesize")):
                print("File exceeded maximum size")
                return redirect(request.url)

            image = request.files["image"]

            if image.filename == "":
                print("Image must have a filename")
                return redirect(request.url)

            if not allowed_image(image.filename):
                print("That image extension is not allowed")
                return redirect(request.url)

            else:
                filename = secure_filename(
                    request.form["fname"] + "." + image.filename.split(".")[1]
                )
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

            print("Image saved")

            return redirect(request.url)

    return render_template("webApp.html")


@app.route("/result", methods=["GET"])
def result():
    return render_template("result.html")


@app.route("/recognize-image", methods=["GET", "POST"])
def recognize_image():
    if request.method == "POST":

        if request.files:

            # if not allowed_image_filesize(request.cookies.get("filesize")):
            #     print("File exceeded maximum size")
            #     return redirect(request.url)

            image = request.files["image"]

            if image.filename == "":
                print("Image must have a filename")
                return redirect(request.url)

            if not allowed_image(image.filename):
                print("That image extension is not allowed")
                return redirect(request.url)

            else:

                save = ""
                filename = secure_filename("test.jpg")
                image.save(os.path.join(save, filename))
                classify_face("test.jpg")
                global crr
                print("Final crr: " + crr)
                return redirect("/result?img="+crr)

            print("Image saved")

            return redirect(request.url)

    return render_template("recognize.html")


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
stream = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = stream.read()
        if not success:
            break
        else:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


if __name__ == "__main__":
    app.run(debug=True, port=8080)


# else:
#                 save = "./"
#                 cv2.imwrite(os.path.join(save, "test.jpg"), image)
#                 image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))