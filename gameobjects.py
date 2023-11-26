import numpy as np 
from snythmodules import *

def dist(x0,y0,x1,y1):
    return ((x0-x1)**2 + (y0-y1)**2)**0.5

class Player:
    def __init__(self, connectionManager:ConnectionManager):
        #Positional Attributes
        self.pos = [0,0]
        self.vel = [0,0]
        self.acc = [0,0]
        self.inAir = False

        # Visual Components
        self.height = 20
        self.width = 10

        #Connectional Attributes
        self.connector_hand = None
        self.is_carrying_connector = False

        #Movement Attributes
        self.jump_height = 13
        self.speed = 3

        #Other
        self.collisions = set()
        self.con_man = connectionManager


    #* === Getters ===
    def getPlayerBottom(self):
        x, y = self.pos[0], self.pos[1]
        return [x+self.width/2, y+self.height]
    
    def getPlayerCenter(self):
        x,y = self.pos[0], self.pos[1]
        h, w = self.height, self.width
        return [x+w/2, y+h/2]
    
    def _getCollisionObjectFromType(self, obj_type):
        for obj in self.collisions:
            if isinstance(obj, obj_type):
                return obj
        return

    #* === Collision Truythys that also update ===

    def isTouchingFloor(self, floor):
        bottom = self.getPlayerBottom()
        pX, pY = bottom[0], bottom[1]
        fX, fY = floor.pos[0], floor.pos[1]
        fW, fH = floor.length, floor.height
        # use a classic rectnagular distance to check collision
        if (fX < pX < fX+fW and fY < pY < fY+fH 
            and self.vel[1]>0):
            self.inAir = False # we're not in air if collided
            # we should also put the player at the top of the platform
            # so we don't accidentally get stuck in the floor
            self.pos[1] -= abs(pY-fY)
            self.vel[1] = 0
            # Update our collisions accordignly
            self.collisions.add(floor)
            return True
        else:
            self.collisions.discard(floor)
            return False

    def isTouchingKnob(self, knob):
        knob_size = knob.jack_size
        if dist(*self.getPlayerCenter(), *knob.pos) < knob_size:
            # update collisions
            self.collisions.add(knob)
            return True
        else:
            #WLOG (from floor), knob
            self.collisions.discard(knob)
            return False
        
    def isTouchingConnector(self, connector:Connector):
        knob = connector.start_jack
        end_knob = connector.to_jack
        if (dist(*self.getPlayerCenter(), *knob.pos) < knob.jack_size or 
            dist(*self.getPlayerCenter(), *end_knob.pos) < knob.jack_size
            ):
            connector.color = 'red'
            self.collisions.add(connector)
            return True
        else:
            connector.color = "green"
            self.collisions.discard(connector)
            return False

    # Actual tr uthy 
    def isInAir(self):
        return self.inAir

    #* === Actions ===
    def stopMoving(self, direction):
        if direction == "left":
            self.vel[0] = 0
        elif direction == "right":
            self.vel[0] = 0
        else:
            pass

    def movePlayer(self, direction, hold=False):
        if direction == "left":
            self.vel[0] = -self.speed
        elif direction == "right":
            self.vel[0] = self.speed
        elif direction == "up" and not hold:
            self._jump()
        elif direction == "down" and not hold:
            self._fall()
        else:
            # this is here since we can pass any direction in, not just avlid ones
            pass

    def _applyGravity(self, force = 0.01):
        if self.isInAir():
            self.acc[1] = force

    def _jump(self):
        if not self.isInAir():
            self.vel[1] = -self.jump_height
            self.inAir = True

    def _fall(self):
        if not self.isInAir():
            self.pos[1] += Floor.height
            self.inAir = True

    def turnKnob(self, direction):
        knob = self._getCollisionObjectFromType(Knob)
        if knob == None: return
        if direction == "left":
            knob.turnLeft()
        elif direction == "right":
            knob.turnRight()
        else:
            print("Invalid Knob Turning Direction")
        
    #* === Connection Stuff ===
    def _beginConnection(self, knob):
        temp = self.con_man.beginConnection(knob)
        if temp is None: return
        self.connector_hand = temp

    def _severConnection(self, connector):
        self.con_man.severConnection(connector)
        self.collisions.discard(connector)

    def _finishConnection(self, knob):
        if self.con_man.placeConnection(knob):
            self.connector_hand = None

    def _dropConnection(self):
        if self.connector_hand is None: return
        self.con_man.dropConnection()
        self.connector_hand = None

    def handleConnectionFromKeyPress(self, key):
        if key == "escape":
            self._dropConnection()
        if key == "d":
            # get if there exists a current connection collision
            connection = self._getCollisionObjectFromType(Connector)
            if connection is None: return
            # remove it
            self._severConnection(connection)
        if key == "space":
            knob = self._getCollisionObjectFromType(Knob)
            if knob is None: return
            if self.connector_hand == None:
                self._beginConnection(knob)
            else:
                self._finishConnection(knob)

    #* === Updates ===
    def updatePositions(self):
        if self.connector_hand != None:
            self.connector_hand.carryEnd(self.pos)


        self.vel[0] += self.acc[0]
        self.vel[1] += self.acc[1]

        self.pos[0] += self.vel[0] + 0.5*self.acc[0]
        self.pos[1] += self.vel[1] + 0.5*self.acc[1]

        self.pos[0] %= 800
        self.pos[1] %= 800

        self._applyGravity(0.5)


    def updateCollisions(self, *items):
        for item in items:
            # checking all possible things it could be colliding with
            if isinstance(item, Module):
                for knob in item.knobs:
                    self.isTouchingKnob(knob)
            elif isinstance(item, Floor):
                self.isTouchingFloor(item)
            elif isinstance(item, Connector):
                # see if it's in either of the jack locations
                self.isTouchingConnector(item)

    #* === Visuals ===
    def Draw(self):
        drawRect(*self.pos, self.width, self.height, fill="blue")


class Floor:
    height = 10
    def __init__(self, position:list, length=800):
        #Misc Attributes
        self.pos = position
        # Visual Attributes
        self.length = length
        self.height= Floor.height
        self.color = 'black'

    #* === Visuals ===
    def Draw(self):
        drawRect(*self.pos, self.length, self.height, fill=self.color)

class Visualizer:
    def __init__(self, position = [600,0]) -> None:
        # Visual Elements
        self.position = position
        self.Xs = np.linspace(0,np.pi*2, self.size[0])
        self.size = [200, 300]

        # Amplitude target and input
        self.target_data = np.array()
        self.input_data = np.array((self.size[0],))

    def _createTarget(self, data):
        self.target_data = data

    def inputData(self, data):
        # we only want the first [size] samples because otherwise it would be unintelligible
        self.input_data = data[:len(self.input_data)]

    def Draw(self):
        pass
        


class Level:
    def __init__(self, level_number):
        self.level_number = level_number
        # To be filled in by rendering 
        self.modules = {Mixer([650,650])}
        self.floors = set()
        # Creating objects in a new level
        self.con_man = ConnectionManager()
        self.player = Player(self.con_man)
        self.output_result = self.createOutput()

        self.renderLevel()

    def renderLevel(self):
        # first add the floors as normal
        for floor_level in range(1,8):
            floor = Floor([0,floor_level*100+10], 800)
            self.floors.add(floor)

        n = self.level_number
        if n == 0:
            self.modules.add(Oscillator([200,0]))
            self.modules.add(Filter([0,0]))
            self.modules.add(Adder([400,0]))
    
    def createOutput(self):
        return
    
    def Draw(self):
        for module in self.modules:
            module.Draw()
        for floor in self.floors:
            floor.Draw()
        
        self.con_man.Draw()
        self.player.Draw()

    def handleKeyPress(self, key):
        self.player.movePlayer(key)
        self.player.handleConnectionFromKeyPress(key)
        if key=="s":
            self._saveData()

    def _saveData(self):
        print("Data Saved (Not really, but will in the future)")
    
    def handleKeyHold(self, key):
        self.player.movePlayer(key, hold=True)
    
    def handleKeyUp(self, key):
        self.player.stopMoving(key)


    def Update(self):
        self.player.updatePositions()
        self.player.updateCollisions(*self.floors,
                                      *self.modules, 
                                      *self.con_man.connections)
        self.con_man.executeConnections()
        
