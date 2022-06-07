import os
import glob

files = glob.glob('src/ExampleFromPaper/images/*.jpg')

for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))