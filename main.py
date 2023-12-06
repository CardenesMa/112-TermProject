from cmu_graphics import *
from GameObjects import Level
import time

#TODO: Create a stepper for the sequencer.
#TODO: Fix bugs with improper use of deleting modules

# Making the splash distinct from the level object is important for clarity
def drawSplash(app):
    w,h = app.width, app.height # for verbosity's sake these are one letter vairable names
    p = 15 # padding
    m = 40  #margin
    r, ir = 30, 20 # radius of jack, inner radius
    cx, cy = w/5, h/4 # location of our circles for the jack
    ox, oy = w*2/3, h/4 # location of the outlet jack
    textsize = 20
    px, py = w/5, h/2-50 # location of our plaeyer head
    pw,ph = 50, 150 # size of our player
    rx,ry = w*2/3, h*3/4-100 # location of our module rect
    rw,rh = 100, 150 # size of our module rect

    drawRect(0,0,w, h,fill=app.background_color) # background
    drawLabel("Eurorack-112", w/2, m, size = 50, bold=True) # Title
    # how to use the dials
    drawCircle(cx,cy,r, fill='red')
    drawLine(cx,cy,cx+r*(2**0.5)/2, cy-r*(2**0.5)/2, fill='white')
    drawCircle(cx,cy,ir)
    drawLabel("Turn dials by", cx,cy-r-p, size=textsize)
    drawLabel("Using 'q' and 'e'", cx,cy+r+p, size=textsize)
    # how to use connectors
    drawRegularPolygon(ox,oy,r,6, fill='grey') 
    drawCircle(ox,oy, ir, fill='green')
    drawCircle(ox+200,oy+100, r, fill='red')
    drawLine(ox+200,oy+100,ox+200+r*(2**0.5)/2, oy+100-r*(2**0.5)/2, fill='white')
    drawCircle(ox+200,oy+100, ir, fill='green')
    drawLine(ox,oy,ox+200,oy+100, fill='green') # line connecting them
    drawLabel("Link plugs by" ,ox,oy-r-p, size=textsize)
    drawLabel("pressing 'space'", ox,oy+r+p,size=textsize)

    # and unlink connectors
    nox, noy = ox, oy+h/5
    drawRegularPolygon(nox,noy,r,6, fill='grey') 
    drawCircle(nox,noy, ir)
    drawCircle(nox+200,noy+100, r, fill='red')
    drawLine(nox+200,noy+100,nox+200+r*(2**0.5)/2, noy+100-r*(2**0.5)/2, fill='white')
    drawCircle(nox+200,noy+100, ir)
    drawLabel("Unlink plugs by " ,nox,noy-r-p, size=textsize)
    drawLabel("pressing 'd'", nox,noy+r+p,size=textsize)

    # Character movement
    drawLabel("Move your character using", px, py-20-p, size=textsize)
    drawLabel("the arrow keys", px, py+ph+p, size=textsize)
    # characyer themself
    drawCircle(px,py,20) # head
    drawLine(px,py,px,py+ph*2/3) # torso
    drawLine(px,py+ph*2/3, px-pw/2, py+ph) # left leg
    drawLine(px,py+ph*2/3, px+pw/2, py+ph) # right leg
    drawLine(px-pw/2, py+ph/4, px+pw/2, py+ph/4) # arms

    # how to make modules
    drawLabel("Place modules by pressing 'a', ", rx, ry-p, size=textsize)
    drawLabel("cycle options with 'q' and 'e', then 'enter'",rx,ry+rh+p, size=textsize)
    drawLabel("Delete with 'backspace'", rx,ry+rh+p*3, size=textsize)
    drawRect(rx,ry, rw,rh, fill='grey', align='top', border="black")
    drawLabel("Module", rx,ry+p)

    # escape dialogue
    drawLabel("Press 'm' to toggle menu", w/2, h-m, size=textsize)





#* == Model ==
def onAppStart(app):
    # This holds all the information for the actual game.
    # All events are passed through here and handled by the class
    app.level = Level()
    # Know if we want to display a splash screen (true at the start)
    app.displaySplash = True

    app.stepsPerSecond = 120

    # visual information
    app.background_color = rgb(242, 229, 215)

    # for delaying key hold
    app.last_time = time.time()
    app.hold_gap = 0.5 # Number of seconds between key press and hold

#* == View ==
def redrawAll(app):
    # make a background
    drawRect(0,0,app.width,app.height, fill = app.background_color)
    # do all the level drawing
    if app.displaySplash:
        drawSplash(app)
    else:
        app.level.Draw()

#* == Controller ==
def onKeyPress(app, key):
    app.last_time = time.time()
    app.level.handleKeyPress(key)
    # Remove the splash screen when 'esc' pressed
    if key == 'm':
        app.displaySplash = not app.displaySplash

def onKeyHold(app, keys):
    if time.time() - app.last_time > app.hold_gap:
        for key in keys:
            app.level.handleKeyHold(key)

def onKeyRelease(app, key): 
    app.level.handleKeyUp(key)

def onStep(app):
    app.level.Update()

def onAppClose(app):
    app.level.Close()

runApp(800,900)