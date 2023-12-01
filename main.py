from cmu_graphics import *
from gameobjects import *

#TODO1: change setFreqs etc so they take an ndarray and get the appropriate values
#tODO4: make a LPF for input signal (https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter)
#TODO2: Make Filter more efficient -- too slow and laggy
#TODO3: Figure out checking algorithm for output
#TODO5: Make it look prettier

#* == Model ==
def onAppStart(app):
    app.levelNumber = 0
    app.levels = [Level(0), Level(1)]
    app.level_index = 0
    app.level = app.levels[app.level_index]

    app.stepsPerSecond = 120

    app.background_color = rgb(242, 229, 215)


#* == View ==
def redrawAll(app):
    # make a background
    drawRect(0,0,app.width,app.height, fill = app.background_color)
    # do all the level drawing
    app.level.Draw()

#* == Controller ==
def onKeyPress(app, key):     
    app.level.handleKeyPress(key)
    if key == 'n':
        app.level_index += 1
        app.level = app.levels[app.level_index]

    if key == "v":
        modules = app.level.modules
        for m in modules:
            if m.title == "Filter":
                m.displayEnvelope = True
def onKeyHold(app, keys):
    for key in keys:
        app.level.handleKeyHold(key)

def onKeyRelease(app, key): 
    app.level.handleKeyUp(key)

def onStep(app):
    app.level.Update()

def onAppClose(app):
    app.level.Close()

runApp(800,800)