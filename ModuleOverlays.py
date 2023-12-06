from cmu_graphics import drawLine, drawLabel, drawRegularPolygon, rgb,drawCircle
import numpy as np

class Slider:
    length = 50
    def __init__(self, *titles) -> None:
        self.titles= titles
        self.pos = [0,0]
        self.selected_index = 0
        assert(len(titles) > 1)
        self.increment = Slider.length // (len(self.titles)-1)
        self.circle_size = 5
        # important so we edon't try to attach anything

    def adjustBounds(self, position):
        self.pos = position

    def moveRight(self):
        if self.selected_index < len(self.titles)-1:
            self.selected_index += 1
        return 1

    def moveLeft(self):
        if self.selected_index > 0:
            self.selected_index -= 1
        return -1

    def Draw(self):
        x,y = self.pos[0], self.pos[1]
        x_0, x_1 = x-Slider.length/2, x+Slider.length/2
        drawLine(x_0,y,x_1,y)
        drawCircle(x_0+self.selected_index*self.increment, y, self.circle_size)
        for title in range(len(self.titles)):
            drawLabel(self.titles[title], x_0+title*self.increment, y-10)
        



class Knob:
    
    def __init__(self, position:list, label: str, 
                method, output=False, initial_value = 0, 
                value_range = (0,1), increment=1,
                slider = None, isInput = False):
        # Visual Information
        self.pos=position
        self.jack_size = 10 # diameter of the hole 
        self.knob_size = 15 # diameter of the turny bit
        self.color = rgb(67, 170, 139)
        self.label = label
        self.slider_length = 20
        self.input = isInput

        self.initial_value = initial_value
        # Functional Information
        self.method = method # the method which it changes
        self.value = initial_value
        self.val_range = value_range # know what the max and min are
        self.increment = increment
        self.isEmpty = True
        self.output = output
        self.slider = slider
        if self.slider != None:
            self.slider.adjustBounds(self.pos)
            self.isEmpty = False
        
        # Store the previous value so we can know if it changed

    def __hash__(self) -> int:
        return hash(self.label)
    
    def __repr__(self) -> str:
        return f"Knob @ {self.pos}"

    def Draw(self):
        if self.isSlider():
            self.slider.Draw()
        else:
            # outside knob bit but the hexagon if output
            if self.isOutput():
                drawRegularPolygon(*self.pos, self.knob_size, 6, fill='grey')
            else:
                drawCircle(*self.pos, self.knob_size, fill = self.color)
                # jack part
                # draw the value and label
                if not self.isInput(): # don't want to read a value if its an input jack 
                    drawLabel(str(self.value), self.pos[0], self.pos[1]-self.jack_size-10)
            # always want the jack present
            drawCircle(*self.pos, self.jack_size, fill = 'black')
            # always want the label, not always the value
            drawLabel(self.label, self.pos[0], self.pos[1]+self.jack_size+10)   

    def turnLeft(self):
        if self.isOutput() or self.isInput(): return
        if self.slider != None:
            self.slider.moveLeft()
        if self.val_range[0] <= self.value - self.increment <= self.val_range[1]:
            self.value -= self.increment
            self.setValue()

    def turnRight(self):
        if self.isOutput() or self.isInput(): return
        if self.slider != None:
            self.slider.moveRight()
        
        if self.val_range[0] <= self.value + self.increment <= self.val_range[1]:
            self.value += self.increment
            self.setValue()


    def isOutput(self):
        return self.output
    
    def isInput(self):
        return self.input

    def isSlider(self):
        return self.slider != None
    
    def getValue(self):
        return self.value

    def setValue(self):
        # this calls the method which 
        # it is linked to, and applies
        # the value which the knob is set to
        #! Important to make sure method returns appropriate value
        output = self.method(np.round(self.value, 1))
        self.value=output

    def connectorRemoved(self):
        self.value = self.initial_value
        self.isEmpty = True

class ConnectionManager:
    def __init__(self):
        self.connections : set[Connector] = set()
        self.temporary_connector = None
        
    def severConnection(self, connector):
        connector.deconstructConnector()
        self.connections.remove(connector)
        self.updateConnections()

    def placeConnection(self, end_knob:Knob):
        # make the jacks know they're full
        if self.temporary_connector.completePlacement(end_knob):
            self.temporary_connector.executeConnection()
            self.connections.add(self.temporary_connector)
            self.temporary_connector = None
            return True
        return False
    
    def updateConnections(self):
        try:
            for connector in self.connections:
                connector.executeConnection()
        except RuntimeError:
            print("Size Changed")
            

    def dropConnection(self):
        # make sure to set the jack as empty
        self.temporary_connector.start_jack.isEmpty = True
        # remove the connector
        self.temporary_connector = None

    # for the initial string-up, creating a new connection
    def beginConnection(self, knob:Knob):
        # handle to make sure it's a valid knob
        if knob.isEmpty:
            self.temporary_connector = Connector(knob)
            return self.temporary_connector
        else: return None

    def Draw(self):
        for connection in self.connections:
            connection.Draw()
        if self.temporary_connector is not None:
            self.temporary_connector.Draw()

    def removeModule(self, module):
        # Get rid of the connection that is attached to the module
        for connection in list(self.connections):
            if (connection.to_jack in module.knobs) or (connection.start_jack in module.knobs):
                self.severConnection(connection)
        self.temporary_connector = None


class Connector:
    ID = 0
    def __init__(self, start_jack:Knob):
        start_jack.isEmpty = False
        self.start_jack = start_jack
        self.to_jack = Knob([0,0], "Default", None)

        self.inMethod = self.start_jack.method
        self.outMethod = None

        # visual information
        self.color = "orange"
        self.loc1 = self.start_jack.pos
        self.loc2 = [0,0]
        self.id = Connector.ID
        Connector.ID += 1

    def __hash__(self) -> int:
        return hash(str(self.id))
    
    def __repr__(self) -> str:
        return f"Connector with ID {self.id}"

    def carryEnd(self, location:list):
        self.loc2 = location

    def getMethods(self):
        if self.start_jack.isOutput():
            outJack = self.start_jack.method
            inJack = self.to_jack.method
        else:
            outJack = self.to_jack.method
            inJack = self.start_jack.method
        return (inJack, outJack)

    def completePlacement(self, end_jack:Knob):
        if not end_jack.isEmpty: return False
        # also, make sure we're connecting 
        # strictly inputs to outputs
        start_type = self.start_jack.isOutput()
        end_type = end_jack.isOutput()
        if start_type == end_type: return
        # we also want to make sure they belong to different objects

        # set the jack to being full
        end_jack.isEmpty = False
        # update where it is plugged in
        self.to_jack = end_jack
        self.outMethod = self.to_jack.method
        self.loc2 = self.to_jack.pos
        # know that the connector is placed successfully
        self.color = "green"
        return True

    def deconstructConnector(self):
        self.start_jack.connectorRemoved()
        self.to_jack.connectorRemoved()

    def executeConnection(self):
        inMethod, outMethod = self.getMethods()
        inMethod(outMethod())

    def Draw(self):
        # for now just a simple straight line
        drawCircle(*self.loc1, 10, fill=self.color)
        drawCircle(*self.loc2, 10, fill=self.color)
        drawLine(*self.loc1, *self.loc2, fill=self.color)
