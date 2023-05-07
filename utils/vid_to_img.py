import cv2
import os


fpath = ["./ytvid.mp4"]
count = 1
for file in fpath:
    video = cv2.VideoCapture(file)
    if video.isOpened() == False:
        print("[INFO] ERROR OPENING VIDEO STREAM OR FILE")
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = video.get(3)   # float `width`
    height = video.get(4)  # float `height`
    dir_name = os.path.dirname(file)
    print("[INFO] {} {} RESOLUTION {}X{}".format(file, dir_name, width, height))
    while video.isOpened():
        (check, frame) = video.read()
        if check:
            print("[INFO] IMAGE {}/{}".format(count, total))
            #frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            fname = os.path.join(dir_name, "images_col/image{}.png".format(count))
            #print("[INFO] fname {} {}".format(fname, os.path.exists(fname)))
            cv2.imwrite(fname, frame)
            count += 1
        else:
            break
    count = 1
