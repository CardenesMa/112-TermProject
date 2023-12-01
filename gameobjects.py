import numpy as np 
from snythmodules import *
import threading
import copy

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
        self.height = 50
        self.width = 20
        self.moving = None
        self.isTurning = False

        #Connectional Attributes
        self.connector_hand = None
        self.is_carrying_connector = False

        #Movement Attributes
        self.jump_height = 7
        self.speed = 5

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
        k_x, k_y  = knob.pos[0], knob.pos[1]
        # use point-in-rectangle checker
        x,y = self.pos[0], self.pos[1]
        w,h = self.width, self.height
        if x < k_x < x+w and y < k_y < y + h:
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
        self.moving = None

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
            return
        self.moving = direction
        self.isTurning = False

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
        if knob.isSlider(): return
        if direction == "q":
            knob.turnLeft()
            self.con_man.updateConnections()
        elif direction == "e":
            knob.turnRight()
            self.con_man.updateConnections()
        else:
            # similar to movement, do nothing
            return        
        self.isTurning = True

    def moveSlider(self, direction):
        knob = self._getCollisionObjectFromType(Knob)
        if knob is None: return
        if not knob.isSlider(): return
        if direction == "e":
            knob.turnRight()
        elif direction == "q":
            knob.turnLeft()
        else:
            pass
        self.con_man.updateConnections()
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
            self.connector_hand.carryEnd(self.getPlayerCenter())


        self.vel[0] += self.acc[0]
        self.vel[1] += self.acc[1]
        
        # Will applying my position twice double the amount of collisions possbile?
        self.pos[0] += 0.5*(self.vel[0] + 0.5*self.acc[0])
        self.pos[1] += 0.5*(self.vel[1] + 0.5*self.acc[1])


        self._applyGravity(0.2)
        
        self.pos[0] += 0.5*(self.vel[0] + 0.5*self.acc[0])
        self.pos[1] += 0.5*(self.vel[1] + 0.5*self.acc[1])

        self.pos[0] %= 800
        self.pos[1] %= 800

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

    def isPlayerInBounds(self, bounds):
        x,y,w,h = bounds

    #* === Visuals ===
    def Draw(self):
        if self.vel[1]<0:
            self._drawMovingUp()
        elif self.moving == 'down':
            self._drawMovingDown()
        elif self.moving == 'left':
            self._drawLeftMovement()
            return
        elif self.moving == 'right':
            self._drawRightMovement()
            return
        else:
            self._drawNotMoving()

    def _drawLeftMovement(self):
        x,y,h,w = self.pos[0], self.pos[1], self.height, self.width
        drawCircle(x+w/2,y+h/4,h/8)#head
        drawLine(x+w/2,y+h/4,  x+w/2,y+3*h/4)#body
        drawLine(x+w/2,y+h/2, x,y+2*h/3)#left arm
        drawLine(x+w/2,y+h/2, x+w, y+h/2) # right arm
        drawLine(x+w/2,y+3*h/4,  x,y+7*h/8 )#left femur
        drawLine(x+w/2, y+3*h/4, x+w,y+h) # right leg
        drawLine(x,y+7*h/8,  x,y+h)# left shin
    def _drawRightMovement(self):
        x,y,h,w = self.pos[0], self.pos[1], self.height, self.width
        drawCircle(x+w/2,y+h/4,h/8)#head
        drawLine(x+w/2,y+h/4,  x+w/2,y+3*h/4)#body
        drawLine(x+w/2,y+h/2, x,y+h/2)#left arm
        drawLine(x+w/2,y+h/2, x+w, y+2*h/3) # right arm
        drawLine(x+w/2,y+3*h/4,  x+w,y+7*h/8 )#right femur
        drawLine(x+w/2, y+3*h/4, x,y+h) # left leg
        drawLine(x+w,y+7*h/8,  x+w,y+h)# right shin

    def _drawMovingUp(self):
        x,y,h,w = self.pos[0], self.pos[1], self.height, self.width
        drawCircle(x+w/2,y+h/4,h/8)#head
        drawLine(x+w/2,y+h/4,  x+w/2,y+3*h/4)#body
        drawLine(x+w/2,y+h/2, x,y+3*h/4)#left arm
        drawLine(x+w/2,y+h/2, x+w, y+3*h/4) # right arm
        drawLine(x+w/2,y+3*h/4,  x+2*w/3,y+4/3*h)#right leg
        drawLine(x+w/2, y+3*h/4, x+w/3,y+4/3*h) # left leg
        
    def _drawMovingDown(self):
        x,y,h,w = self.pos[0], self.pos[1], self.height, self.width
        drawCircle(x+w/2,y+h/4,h/8)#head
        drawLine(x+w/2,y+h/4,  x+w/2,y+3*h/4)#body
        drawLine(x+w/2,y+h/2, x,y+h/4)#left arm
        drawLine(x+w/2,y+h/2, x+w, y+h/4) # right arm
        drawLine(x+w/2,y+3*h/4,  x+w,y+3*h/4)#right leg
        drawLine(x+w/2, y+3*h/4, x,y+3*h/4) # left leg

    def _drawNotMoving(self):
        x,y,h,w = self.pos[0], self.pos[1], self.height, self.width
        drawCircle(x+w/2,y+h/4,h/8)#head
        drawLine(x+w/2,y+h/4,  x+w/2,y+3*h/4)#body
        drawLine(x+w/2,y+h/2, x,y+h/2)#left arm
        drawLine(x+w/2,y+h/2, x+w, y+h/2) # right arm
        drawLine(x+w/2,y+3*h/4,  x,y+h )#left leg
        drawLine(x+w/2, y+3*h/4, x+w,y+h) # right leg


class Floor:
    height = 5
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

# class TargetOutput:
#     def __init__(self, output_audiomethod, pos = [600,0]):
#         self.method = output_audiomethod
#         self.pos =pos
#         self.size= (200,100)
#         self.duration = 1

#         self.stream = PA.PyAudio().open(
#             44100,
#             1,
#             PA.paInt16,
#             output=True,
#             frames_per_buffer=256
#         )

#         self.audio_cache = np.array(44100*self.duration)

#         self.check_values = (self.pos[0],self.pos[1], self.size[0]/2,self.size[1])
#         self.play_values = (self.pos[0]+self.size[0]/2, self.pos[1], self.size[0]/2, self.size[1])

#     def play(self):
#         while True:
#             samples = self.audio_cache
#             samples = np.array(samples).astype(np.int16)
#             self.stream.write(samples.tobytes())



#     def click(self, player:Player, mixer:Mixer, mixer_thread):
#         x,y = player.pos[0],player.pos[1]
#         cx, cy, cw, ch = self.check_values
#         px, py, pw, ph = self.play_values
#         if cx < x < cx+cw and cy<y<cy+ch:
#             return self.check(mixer.input_generator)
            
#         elif px < x < px+pw and py<y<py+ph:
#             mixer_thread.join()
#             self.play()
#             mixer_thread.start()
#         return False
        
#     def check(self, other):
#         # # since we can't store both generators at the same time
#         # # (as they are aliased and can't be copied), l ook at their respective FFTs
#         # # Then we can determine the difference of the FFTs
#         # N = 1024
#         # other_transform = np.fft.fft()
#         # answer_transform = np.fft.fft(())
#         # xs = np.linspace(0,1,256)
#         # plt.plot(xs, other_transform)
#         # plt.plot(xs, answer_transform)
        
#         return False

#     def Draw(self):
#         cx, cy, cw, ch = self.check_values
#         px, py, pw, ph = self.play_values
#         drawRect(cx,cy,cw,ch, fill="red")
#         drawRect(px,py,pw,ph, fill='green')

#         drawLabel("Check", cx+cw/2,cy+ch/2)
#         drawLabel("Play", px+pw/2, py+ph/2)

#     def setData(self, data):
#         self.audio_cache = np.roll(self.audio_cache, len(data))
#         self.audio_cache[:len(data)] = data


class Level:
    def __init__(self, level_number=0):
        
        # self.level_number = level_number
        # mixer information
        self.mixer = Mixer([0,0])
        self.output_result = self.createOutput()
        self.con_man = ConnectionManager()
        # self.target = TargetOutput(self.output_result)
        #CRAZY SOLUTION: pass the manager into the playsound method so that 
        self.audioThread= threading.Thread(target=self.mixer.playSound, args=[self.con_man])
        # To be filled in by rendering
        self.modules = {self.mixer}
        self.floors = set()
        # Creating objects in a new level
        self.player = Player(self.con_man)
        self.isSolved = False

        self.block_sizes = [200,300]
        # since the player will be able to add modules there
        self.temporary_module = None
        self.temporary_module_loc = [0,0]
        # we will iterate between these to choose what 
        self.module_types = [Oscillator, LFO, Adder, LPF, Filter]
        self.module_types_loc = 0
        self.isPlacingModule = False
        

        self.renderLevel()

    def renderLevel(self):
        # first add the floors as normal
        for floor_level in range(1,8):
            length=800
            floor = Floor([0,floor_level*100+25], length)
            self.floors.add(floor)

        # Possibly no logner needed
        # n = self.level_number
        #     self.modules.add(Oscillator([200,0]))
        #     self.modules.add(LPF([200,400]))
        #     self.modules.add(LFO([0,200], 10))
        
        # Start the trhead of our audio
        self.audioThread.start()

    
    def createOutput(self):
        return self.mixer.audio_cache
    
    def Draw(self):
        for module in self.modules:
            module.Draw()
        # also draw our temporary module if there
        if self.temporary_module != None:
            self.temporary_module.drawTemporary()
        # draw all the floors on top
        for floor in self.floors:
            floor.Draw()
        
        # then overlay the connectors
        self.con_man.Draw()
        # self.target.Draw()
        self.player.Draw()


    def handleKeyPress(self, key):
        self.player.movePlayer(key)
        self.player.turnKnob(key)
        self.player.handleConnectionFromKeyPress(key)

        # if key == 'space':
        #     if self.target.click(self.player,self.mixer, self.audioThread):
        #         self.isSolved = True
        self.player.moveSlider(key)
        # if key=="s":
        #     print("saved")
        #     self._saveData()

        # begin placing the module with a
        if key == 'a' and not self.isPlacingModule:
            self.isPlacingModule = True
            self.temporary_module = self.module_types[self.module_types_loc](self.temporary_module_loc)
            
        # look through modules left with s
        if key == 'q' and self.isPlacingModule:
            t = self.module_types # for verbosity
            self.module_types_loc -= 1
            self.module_types_loc %= len(t)
            l = self.module_types_loc
            self.temporary_module = t[l](self.temporary_module_loc)
        
        # look through modules right with d
        if key == 'e' and self.isPlacingModule:
            t = self.module_types # for verbosity
            self.module_types_loc += 1
            self.module_types_loc %= len(t)
            l = self.module_types_loc
            self.temporary_module = t[l](self.temporary_module_loc)

        
        if key == 'enter' and self.isPlacingModule:
            # we are no longer going to be placing a module
            self.isPlacingModule = False
            # update to make sure the loc is correct
            self.updateTempModuleLoc()
            # create a new module that isn't temporary
            new_module= self.module_types[self.module_types_loc](self.temporary_module.pos)
            # add that new module to our set
            self.modules.add(new_module)
            # 
            self.temporary_module = None
        

    def _saveData(self):
        self.target.setData(self.createOutput())
    
    def handleKeyHold(self, key):
        self.player.movePlayer(key, hold=True)
        self.player.turnKnob(key)
    
    def handleKeyUp(self, key):
        self.player.stopMoving(key)
        self.player.isTurning = False


    def Update(self):
        self.player.updatePositions()
        self.player.updateCollisions(*self.floors,
                                      *self.modules, 
                                      *self.con_man.connections)
        if self.temporary_module != None:
            self.updateTempModuleLoc()
        
    def updateTempModuleLoc(self):
        # round the position of the player to the nearest block place
        x , y = self.player.pos[0], self.player.pos[1]
        bw,bh = self.block_sizes[0], self.block_sizes[1]
        # want to round the module lcoation to nearest block
        temp_pos = [int(bw*np.floor(x/bw)), int(bh*np.ceil(y/bh)-bh)]
        # make sure it is valid
        if self.isValidPlacingPos(temp_pos): 
            self.temporary_module.pos = temp_pos
    
    def isValidPlacingPos(self, pos):
        for module in self.modules:
            if module.pos == pos:
                return False
        return True

    def Close(self):
        self.mixer.close()
        self.audioThread.join()
        print("Closed")

    def isComplete(self):
        return self.isSolved