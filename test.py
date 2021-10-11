import pickle as p
import config
import os
from tool import img_path_to_GEI
import numpy as np
from model.models import RandomForestClassification
from scipy.misc import imread
from feature.hog import flatten
import PySimpleGUI as sg

sg.theme('BluePurple')

test = "testData"
initial_folder = "%s\%s" % (config.Project.project_path, test)
validation_x = []
img_path = ''
predict_file = "%s\model\predict.pickle" % config.Project.project_path
modell = p.load(open(predict_file, 'rb'))

layout = [[sg.Text('Select an image to be classified')],
          [sg.Input(), sg.FileBrowse(initial_folder = initial_folder)],
          [sg.Image( key = '-IMAGE-')],
          [sg.Button('Show Image'), sg.Submit(), sg.Cancel()]]

window = sg.Window('Age Estimation by Gait', layout)

while True:
    event, values = window.read()

    if event == 'Cancel' or event == sg.WIN_CLOSED:
        break

    if event == 'Show Image':
        window['-IMAGE-'].update(filename = values[0], size = (500, 200))

    if event == 'Submit':
        img_path = values[0]
        values = []
        img_dir = []
        im = imread(img_path)
        img_path = ''
        img_dir.append(im)
        data = img_path_to_GEI(img_dir)
        img_dir = []
        validation_x.append(data)
        validation_feature_x = [flatten(x) for x in validation_x]
        validation_x = []


        predict_y = modell.predict(validation_feature_x)
        validation_feature_x = []
        age_group_ID = predict_y[0]
        predict_y = []
        age_group = ''

        if age_group_ID == 'A':
            age_group = '0 - 5'
        elif age_group_ID == 'B':
            age_group = '6 - 10'
        elif age_group_ID == 'C':
            age_group = '11 - 15'
        elif age_group_ID == 'D':
            age_group = '16 - 20'
        elif age_group_ID == 'E':
            age_group = '21 - 30'
        elif age_group_ID == 'F':
            age_group = '31 - 40'
        elif age_group_ID == 'G':
            age_group = '41 - 50'
        elif age_group_ID == 'H':
            age_group = '51 - 60'
        elif age_group_ID == 'I':
            age_group = 'Above 60'

        age_group_ID = ''
        sg.popup('Estmated age group is', age_group)
        age_group = ''
    
window.close()


'''
structure = [[sg.Text('Estimated age group is '), sg.Text(key = '-OUTPUT-')]]
popup = sg.Window('Result',structure)

while True
e, val = popup.read()
popup.close()
'''