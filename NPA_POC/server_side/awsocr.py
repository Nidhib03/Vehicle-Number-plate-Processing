import boto3
import os
import cv2
import numpy as np
import importlib.util
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from app import app
from flask import request, jsonify
from werkzeug.utils import secure_filename
import re
from datetime import datetime

client = boto3.client('rekognition')

scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name("novice-342709-9a8f64515cff.json", scopes)
file = gspread.authorize(credentials) 
sheet = file.open("NPAws").get_worksheet(5)
tlist = []

@app.route('/input_image', methods=['POST'])
def upload_video():
    crop_file = request.files['file']
    
    print("file>>>>>>>>>>>", crop_file)
    if request.method == "POST":
        if 'file' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
        crop_file = request.files['file']
        if crop_file.filename == '':
            resp = jsonify({'message' : 'No file selected for uploading'})
            resp.status_code = 400
            return resp

        if crop_file:
            filename = secure_filename(crop_file.filename)
            print("file name =====================", filename)

            crop_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resp = jsonify({"filename": filename, "Status":200})
            print("resp!!!!!!!!!!!!!!!",resp)
            resp.status_code = 200
            
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                                                   
            # Videos_Time = re.sub('\D', '', filename) #remove all things exclude number
            # START_Videos_Time = str(Videos_Time[0:4]+'-'+Videos_Time[4:6]+'-'+Videos_Time[6:8]+' '
            #     +Videos_Time[8:10]+":"+Videos_Time[10:12]+":"+Videos_Time[12:14]) 
            # print("START_Videos_Time ::::::::::::",START_Videos_Time)

            # def convert(seconds):
            #     seconds = seconds % (24 * 3600)
            #     print("seconds ::::::::",seconds)
            #     hour = seconds // 3600
            #     seconds %= 3600
            #     minutes = seconds // 60
            #     seconds %= 60
            #     return seconds,minutes,hour

            # sec = 0               
            # date_time_obj = datetime.datetime.strptime(START_Videos_Time, '%Y-%m-%d %H:%M:%S')
            # sec = round(sec, 2)
            # Full_TIme = convert(sec)
            # print("data time",date_time_obj + datetime.timedelta(seconds=Full_TIme[0],minutes=Full_TIme[1],hours=Full_TIme[1]))
                                    
            l = len(sheet.col_values(1)) 
            l += 1
            nm = []
            imageSource=open('Crop/'+filename,'rb')
            print('imageSource"""""""""',imageSource)
            response=client.detect_text(Image={'Bytes': imageSource.read()})
            textDetections = response['TextDetections']
            print ('plate Detected\n----------------------------------')
            
            for text in textDetections:
                txt = text['DetectedText']
                # print ('Detected text: ' + txt)
                                    
                r = bool(re.search(r'\d{4}$',txt))
                t = bool(re.search(r'^[A-Z]{2}',txt))
                # te = len(txt)
                
                # if t==4 or t==5 and 'Id' in text:
                if t and (len(txt)==5 or len(txt)==4) or r and 'Id' in text:
                    if text['Type']=='LINE':
                        nm.append(txt)
                        print("text]]]]]]]]]]]]]]]] ", nm)                       
            npa = []
            if len(nm)==1:
                nu = nm[0]
                n = nu.strip('IND')
                print("CAR Number plate Recognised:  ", n)
                npa.append(n)
                if n not in tlist:              
                    tlist.append(n)
                    npa.insert(0, str(dt_string))  #+ datetime.timedelta(seconds=Full_TIme[0],minutes=Full_TIme[1],hours=Full_TIme[1])))
                    print("Sheet Entry:::::::::",npa)
                    sheet.update('A'+str(l),[npa])
                
            elif len(nm)>=1:
                nu = str(nm[0])+str(nm[1])
                n = nu.strip('IND')
                print('BIKE Number plate Recognised:  ',n) 
                npa.append(n)
                if n not in tlist:
                    tlist.append(n) 
                    npa.insert(0, str(dt_string))  #+ datetime.timedelta(seconds=Full_TIme[0],minutes=Full_TIme[1],hours=Full_TIme[1])))
                    print("Sheet Entry:::::::::",npa)
                    sheet.update('A'+str(l),[npa])
            else:
                print("Detected but Not Recognised............!!!")
                pass
            print("tlist", tlist, '\n',len(tlist))
          
            return resp
        return resp 
    return 'file' 

if __name__=="__main__":
    # context = ('/etc/letsencrypt/live/npr.mylionsgroup.com/cert.pem','/etc/letsencrypt/live/npr.mylionsgroup.com/privkey.pem')
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 8011)))
            # , ssl_context=context)