from cmu_graphics import *
from gameobjects import *

#TODO: Make Filter more efficient -- too slow and laggy
#TODO: Make sprite 
#TODO: Make it look prettier
#For After MVP:
# Put graphics in it's own thread and do audio

#* == Model ==
def onAppStart(app):
    app.levelNumber = 0
    app.level = Level(0)


#* == View ==
def redrawAll(app):
    app.level.Draw()

#* == Controller ==
def onKeyPress(app, key):
    app.level.handleKeyPress(key)

def onKeyHold(app, keys):
    for key in keys:
        app.level.handleKeyHold(key)

def onKeyRelease(app, key):
    app.level.handleKeyUp(key)

def onStep(app):
    app.level.Update()

runApp(800,800)