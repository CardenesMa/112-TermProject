import numpy as np
from cmu_graphics import *

def generator(start, step):
    s = start
    while True:
        yield s
        s += step


class Module:
    def __init__(self):
        #Visual Information
        self.pos = [0,0]
        self.size = [200,400]
        self.background = 'lavender'
        self.title = "Default Module"

        # IO information
        self.sample_rate = 44100
        self.bit_depth = np.int16
        self.sample_size = 256

        # Audio Data 
        self.audio_cache = np.zeros((self.sample_size,))
        self.input_generator = generator(0,1)

        # create knobs as required
        self.knobs = set()

    def _attachKnobs(self):
        pass

    # Visualization
    def Draw(self):
        # background stuff
        drawRect(*self.pos, *self.size, fill = self.background, border="black")
        drawLabel(self.title, self.pos[0]+self.size[0]/2, self.pos[1]+10)
        # draw all the knobs
        for knob in self.knobs:
            knob.Draw()

    # I/O
    def Input(self, generator):
        self.input_generator = generator

    def Output(self):
        return next(self.input_generator)


class Oscillator(Module):
    def __init__(self, topleft, frequency = 440, amplitude = 1000):
        super().__init__()
        #overrides
        self.pos = topleft
        self.title = "'V'CO"
        #locals
        self.freq = frequency
        self.amp = amplitude
        self.active = False

        # Put knobs in our set (from parent)
        self._attachKnobs()

    # Initiation methods
    def _attachKnobs(self):
        # Create an instance of each knob that should be on the oscillator
        x, y = self.pos[0], self.pos[1]
        freqKnob = Knob(
            [x+50,y+100],
            "F",
            self.setFrequency,
            False,
            440,
            (50,1000),
            2
        )
        ampKnob = Knob(
            [x+100,y+100],
            "A",
            self.setAmplitude,
            False,
            0,
            (0,(2**15-1)/2),
            10
        )
        activeKnob = Knob(
            [x+150,y+100],
            "1/0",
            self.setActive,
            False,
            0,
            (0,1),
            1
        )
        outputJack = Knob(
            [x+150,y+200],
            "Out",
            self.Output,
            True
        )
        self.knobs.add(freqKnob)
        self.knobs.add(ampKnob)
        self.knobs.add(activeKnob)
        self.knobs.add(outputJack)

    def Output(self):
        # Generator method for sin wave (other waves to come later)
        increment = (2*np.pi*self.freq)/self.sample_rate 
        return (np.sin(k)*self.amp for k in generator(0,increment))

    #* === Setters ===
    def setAmplitude(self, amount):
        amount = next(amount)
        if amount < (2**15-1)/2: self.amp = amount
    
    def setFrequency(self, amount):
        amount  = 440 #! This is static for now! 
        if 0 < amount < 4000: self.freq = amount

    def setActive(self, active):
        active= next(active)
        self.active = active>0

    #* === Getters ===
    def getAmplitude(self):
        return self.amp

    def getFrequency(self):
        return self.freq

    def isActive(self):
        return self.active
    

class Mixer(Module):
    def __init__(self, position):
        super().__init__()
        self.title = "Mixer"
        self.pos= position
        self.size=[200,100]
        # Locals
        self._attachKnob()
        
        # Audio Output Information (Can't use until MVP)
        self.stream = None
        self.isActive = False

    def _attachKnob(self):
        x,y = self.pos[0], self.pos[1]
        input_knob = Knob(
            [x+50, y+50],
            "In",
            self.Input,
            False
        )
        self.knobs.add(input_knob)

    def Output(self):
        # This will be done later when we have MVP (audio segment here)
        # Currenty holds the total output information
        return self.audio_cache

    def Input(self, generator):
        # update our cache with the appropriate stuff
        for i in range(self.sample_size):
            self.audio_cache[i] = next(generator)
    

        

class Adder(Module):
    def __init__(self, position):
        super().__init__()
        self.pos = position
        self.title = "Adder"

        self._attachKnobs()

        self.generator1 = None
        self.generator2 = None

    def _attachKnobs(self):
        x,y = self.pos[0], self.pos[1]
        knobA = Knob(
            [x+50,y+50],
            "A",
            self.In1,
            False,
            0,
            (0,0),
            0
        )
        knobB = Knob(
            [x+100,y+50],
            "B",
            self.In2,
            False,
            0,
            (0,0),
            0
        )

        outKnob = Knob(
            [x+160, y+200],
            "Out",
            self.Output,
            True
        )
        self.knobs.add(knobA)
        self.knobs.add(knobB)
        self.knobs.add(outKnob)

    def In1(self, generator):
        self.generator1= generator
    def In2(self, generator):
        self.generator2= generator

    def Output(self):
        return (self._limitedValues() for i in generator(0,1))

    def _limitedValues(self):
        if self.generator1 is not None and self.generator2 is not None:
            a = next(self.generator1)
            b = next(self.generator2)
            res = a + b
            if res > (2**15-1)/2: res = (2**15-1)//2
            elif res < -(2**15-1)/2: res = -(2**15-1)//2
            return res
        return 0


class Filter(Module):
    def __init__(self, position, 
                 attack=0.5, decay = 0.5,
                 sustain = 0.5, release = 1,
                 seconds = 4):
        super().__init__()
        self.pos = position
        self.title = "Filter"
        self.duration = seconds

        # the number of samples which holds the asdr as needed
        self.attack_samples = int(self.sample_rate * attack)
        self.decay_samples = int(self.sample_rate * decay)
        self.sustain_samples = int((self.duration-decay-release-attack)*self.sample_rate)
        self.sustain_amount = sustain
        self.release_samples = int(self.sample_rate * release)
        # total number of samples in the filter
        self.num_samples = self.duration*self.sample_rate
        # this holds the actual data of inputting
        self.audio_cache = np.zeros(self.num_samples)
        # is the values of the envelope itself at that length
        self.envelope = np.zeros(self.num_samples)
        
        self._attachKnobs()
        self._createEnvelope()

    def _createEnvelope(self):
        # this is used since too verbose otherwise
        Ea = self.attack_samples
        Es = self.sustain_samples
        Ed = self.decay_samples
        Er = self.release_samples
        # attack
        self.envelope[:Ea] = np.linspace(0,1,Ea)
        # decay
        self.envelope[Ea:Ea+Ed] = np.linspace(1,self.sustain_amount, Ed)
        # sustain
        self.envelope[Ea+Ed:Ea+Ed+Es] = self.sustain_amount
        # release
        self.envelope[-Er:]=np.linspace(self.sustain_amount,0,Er)


    def _attachKnobs(self):
        attKnob = Knob(
            [40, 100],
            "Attack",
            self.setAttack,
            False,
            0,
            (0,2),
            0.1
        )
        decayKnob = Knob(
            [80, 100],
            "Decay",
            self.setDecay,
            False,
            0,
            (0,2),
            0.1
        )
        sustainKnob = Knob(
            [120, 100],
            "Sustain",
            self.setSustain,
            False,
            1,
            (0,1),
            0.1
        )
        releaseKnob = Knob(
            [160, 100],
            "Release",
            self.setRelease,
            False,
            0,
            (0,2),
            0.1
        )
        outputKnob = Knob(
            [160,200],
            "Out",
            self.Output,
            True
        )
        self.knobs.add(attKnob)
        self.knobs.add(decayKnob)
        self.knobs.add(sustainKnob)
        self.knobs.add(releaseKnob)
        self.knobs.add(outputKnob)
    
    # want to fill my audio packet up to the number of samples
    # for me to perform the 
    def Input(self, generator):
        self.input_generator = generator
        
    def Output(self):
        # first get the information from the generator into our audio,
        # then process the audio information as needed, 
        # and return a generator which accesses our cached data
        for sample in range(self.num_samples):
            self.audio_cache[sample] = next(self.input_generator) # fill our input data

        # we can create a numpy array of samples from 0-1 and then
        # multiply the envelope
        index = 0 
        while True:
            # since we always want to update once 
            # we've finished looking through the 
            # filter, do it when we reset our index
            if index == 0:
                signal = self._applyFilter()
            # generates the next bit in our smaple
            yield signal[index]
            # keep looping through the signal
            index += 1
            # once we've gone through the whole list
            # restart again
            index %= self.num_samples

    def _applyFilter(self)->np.ndarray:
        final_signal = self.audio_cache * self.envelope
        # normalize it to approrpriate means
        final_signal /= np.max(np.abs(final_signal))
        return final_signal

    #* Setters
    def setAttack(self, amount):
        self.attack = amount
    
    def setDecay(self, amount):
        self.decay = amount
    
    def setSustain(self, amount):
        self.sustain = amount
    
    def setRelease(self, amount):
        self.release = amount
    

class Knob:
    def __init__(self, position:list, label: str, method, output=False, initial_value = 0, value_range = (0,1), increment=1):
        # Visual Information
        self.pos=position
        self.jack_size = 10 # diameter of the hole 
        self.knob_size = 15 # diameter of the turny bit
        self.color = "red"
        self.label = label

        # Functional Information
        self.method = method # the method which it changes
        self.value = initial_value
        self.val_range = value_range # know what the max and min are
        self.increment = increment
        self.isEmpty = True
        self.output = output
    
    def __hash__(self) -> int:
        return hash("k" + str(self.pos))
    
    def __repr__(self) -> str:
        return f"Knob @ {self.pos}"

    def Draw(self):

        # outside knob bit but the hexagon if output
        if self.isOutput():
            drawRegularPolygon(*self.pos, self.knob_size, 6, fill='grey')
        else:
            drawCircle(*self.pos, self.knob_size, fill = self.color)
        # jack part
        drawCircle(*self.pos, self.jack_size, fill = 'black')
        # draw the value and label
        drawLabel(self.label, self.pos[0], self.pos[1]-self.jack_size-10)
        drawLabel(str(self.value), self.pos[0], self.pos[1]+self.jack_size+10)

    def turnLeft(self):
        self.value -= self.increment
        self.setValue()
    def turnRight(self):
        self.value += self.increment
        self.setValue()

    def isOutput(self):
        return self.output

    def setValue(self):
        # this calls the method which 
        # it is linked to, and applies
        # the value which the knob is set to
        self.method(self.value)

class ConnectionManager:
    def __init__(self):
        self.connections : set[Connector] = set()
        self.temporary_connector = None
        
    def executeConnections(self):
        for connector in self.connections:
            inMethod, outMethod = connector.getMethods()
            inMethod(outMethod())

    def severConnection(self, connector):
        connector.deconstructConnector()
        self.connections.remove(connector)

    def placeConnection(self, end_knob:Knob):
        # make the jacks know they're full
        if self.temporary_connector.completePlacement(end_knob):
            self.connections.add(self.temporary_connector)
            self.temporary_connector = None
            return True
        return False

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
        self.start_jack.isEmpty = True
        self.to_jack.isEmpty = True

    def Draw(self):
        # for now just a simple straight line
        drawCircle(*self.loc1, 10, fill=self.color)
        drawCircle(*self.loc2, 10, fill=self.color)
        drawLine(*self.loc1, *self.loc2, fill=self.color)
