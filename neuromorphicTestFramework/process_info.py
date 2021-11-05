
import os
import json

def main():

    folderName = 'folder_exp'
    ensaios =[]

    onlyfiles = [filename for count, filename in enumerate(sorted(os.listdir(folderName)))]
    for file in onlyfiles:
        f = open(folderName + "/" + file, "r")
        Py_object = json.load(f)
        f.close()
        ensaios.append(Py_object)


    for ensaio in ensaios:
        print(ensaio)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
