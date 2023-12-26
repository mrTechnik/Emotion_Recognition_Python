import cv2
from tkinter import Tk, Button, PhotoImage, filedialog, Frame, TOP, Canvas, LEFT, RIGHT, BOTTOM, BOTH, NW, NE
from PIL import Image, ImageTk
from keras.models import load_model
from keras_preprocessing.image.utils import img_to_array
from numpy import sum as npsum, expand_dims
from matplotlib.pyplot import figure, close
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from threading import Thread


def close_on_window():
    global running
    running = False


class App:
    class_labels = ('Angry', 'Happy', 'Neutral', 'Sad', 'Surprise')
    preds = [0, 0, 0, 0, 0]
    count = 100
    file = 'EmotiCon' + datetime.now().strftime('%Y-%m-%d_%H_%M') + '.txt'
    lines = []
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model('EmotionDetectionModel.h5')

    def __init__(self, window, window_name, window_logo_path, fps):
        self.window = window
        self.window.title(window_name)
        self.window.geometry("400x200")
        self.window.iconphoto(
            False,
            PhotoImage(file=window_logo_path)
        )
        self.window.configure(
            bg='#111111',
            bd=0
        )
        self.window.protocol("WM_DELETE_WINDOW", close_on_window)
        self.window.resizable(0, 0)
        self.fps = fps

        self.dynamicrecbutton = Button(
            self.window,
            activebackground='#ffffff',
            bg='#aaaaaa',
            text='Dynamic recognition', height=2, width=16,
            command=lambda: self.video_recognise(video_source=0)
        )

        self.dynamicrecbutton.place(x=(self.window.winfo_width() // 2) - 56, y=20)

        self.staticrecbutton = Button(
            self.window,
            activebackground='#ffffff',
            bg='#aaaaaa',
            text='Static recognition',
            height=2,
            width=16,
            command=lambda: self.video_recognise(video_source=filedialog.askopenfilename())
        )

        self.staticrecbutton.place(x=(self.window.winfo_width() // 2) - 56, y=80)

        self.exitbutton = Button(
            self.window,
            activebackground='#ffffff',
            bg='#aaaaaa', text='Exit',
            command=lambda: self.window.destroy(),
            height=2,
            width=6
        )

        self.exitbutton.place(x=(self.window.winfo_width() // 2) - 21, y=140)

        self.window.mainloop()

    def video_recognise(self, video_source):

        self.dynamicrecbutton.destroy()
        self.staticrecbutton.destroy()
        self.exitbutton.destroy()

        # open video
        self.window.geometry("1000x800")
        self.window.resizable(0, 0)
        self.video = VideoCapture(self.window, video_source)

        self.image_frame = Frame(
            self.window,
            width=self.window.winfo_width(),
            height=self.window.winfo_height() // 2,
            bg='#111111'
        )

        self.image_frame.pack(side=TOP, expand=True)

        self.canvas = Canvas(
            self.image_frame,
            width=self.window.winfo_width() // 2,
            height=self.window.winfo_height() // 2
        )

        self.canvas.pack(side=LEFT, expand=True)

        self.canvas_grey = Canvas(
            self.image_frame,
            width=self.window.winfo_width() // 2,
            height=self.window.winfo_height() / 2
        )

        self.canvas_grey.pack(side=RIGHT, expand=True)

        self.diagram_frame = Frame(
            self.window,
            width=self.window.winfo_width(),
            height=self.window.winfo_height() // 2
        )

        self.diagram_frame.pack(side=BOTTOM, expand=True)

        self.curves = {emotion: [0. for _ in range(self.count)] for emotion in self.class_labels}
        self.fig_big = figure()
        ax_big = self.fig_big.add_subplot(211)
        ax_big.set_ylim([0, 1])

        for i in self.class_labels:
            buf_line = ax_big.plot(self.curves[i], label=i)
            self.lines.append(buf_line)

        ax_big.legend()
        bar2 = FigureCanvasTkAgg(self.fig_big, self.diagram_frame)
        bar2.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=True)

        self.fig = figure()
        ax = self.fig.add_subplot(211)
        ax.set_ylim([0, 1])

        self.line, = ax.plot(self.class_labels, self.preds, lw=2)
        bar1 = FigureCanvasTkAgg(self.fig, self.diagram_frame)
        bar1.get_tk_widget().pack(side=RIGHT, fill=BOTH, expand=True)

        self.delay = int(1000 / self.fps)
        self.update()

        self.window.mainloop()

    def update(self):
        ret, frame = self.video.get_frame()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)
        end_gray = gray

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            end_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if npsum([roi_gray]) != 0:
                roi = expand_dims(img_to_array(roi_gray.astype('float') / 255.0), axis=0)
                self.preds = self.classifier.predict(roi)[0]

                label = self.class_labels[self.preds.argmax()]
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No Face Found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        if ret:
            if running:
                self.photo = ImageTk.PhotoImage(
                    image=Image.fromarray(frame).resize((self.window.winfo_width() // 2,
                                                        self.window.winfo_height() // 2)))

                self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

                self.photo_grey = ImageTk.PhotoImage(
                    image=Image.fromarray(end_gray).resize((self.window.winfo_width() // 2,
                                                            self.window.winfo_height() // 2)))

                self.canvas_grey.create_image(self.window.winfo_width() // 2, 0, image=self.photo_grey, anchor=NE)

                my_thread = Thread(target=self.write_in_file, args=(self.file, str(self.preds)))
                my_thread.start()

                self.line.set_ydata(self.preds)

                for k, i in enumerate(self.class_labels):
                    self.curves[i] = self.curves[i][1:]
                    self.curves[i].append(self.preds[k])
                    self.lines[k][0].set_ydata(self.curves[i])
                    self.lines[k][0].set_label(self.preds[k])

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                self.fig_big.canvas.draw()
                self.fig_big.canvas.flush_events()

            else:
                self.window.destroy()
                close(self.fig)
                close(self.fig_big)

        self.window.after(self.delay, self.update)

    def write_in_file(self, filename, data):
        with open(filename, 'a') as f:
            f.write(data + '\n')


class VideoCapture:
    def __init__(self, window, video_source=0):
        self.window = window
        # start video
        self.video = cv2.VideoCapture(video_source)
        if not self.video.isOpened():
            raise ValueError("Unavailable video source")

        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            exit()


running = True


def main():
    try:
        App(Tk(), 'EmotiCon', 'icon.png', 60)
    except BaseException as be:
        print(be.__class__.__name__, be.args)
    #compileall.compile_file('emotion_recognise_class.py')


if __name__ == '__main__':
    main()


