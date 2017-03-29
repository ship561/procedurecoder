from flask import render_template, request
from src import app
import pandas as pd
import numpy as np
import pickle
import drchrono_predict as dpred
import csv


@app.route('/')
def root():
    return render_template("input.html", title = 'Procedure Coder')


@app.route('/input')
def cpt_input():
    return render_template("input.html")


@app.route('/output', methods=["GET", "POST"])
def cpt_output():
    #pull 'birth_month' from input field and store it
    doc = request.args.get('birth_month')
    pred = dpred.main(doc)
    print pred

    birth = []
    for i in range(0, pred.shape[0]):
        birth.append(dict(index = i+1, code=pred.iloc[i]['code'],
                          description=pred.iloc[i]['description']))
    return render_template("output.html", title = 'Procedure Coder',
                           births = birth, the_result = doc)

@app.route('/keep')
def store_cpts():
    docs = request.args.get('doc')
    cpt_codes = request.args.get('recs')
    
    xxx = ';'.join(request.args.getlist('check'))
    with open("temp_output.csv", 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(["doctors note", "CPT codes"])
        writer.writerow([docs, xxx])
    
    print "here is the xxx ", xxx, " docs ", docs, "cpt codes", cpt_codes
    return render_template("input.html", title = 'Procedure Coder')
        
@app.route('/slides')
def get_slides():
    return render_template("demo_slides.html", title = 'Demo Slides')

@app.route('/secretslides')
def get_secretslides():
    return render_template("demo_slides_secret.html", title='Demo Slides')

@app.route('/contact')
def get_contact():
    return render_template("contact.html", title= 'Contact me')
