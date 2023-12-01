import numpy as np
from cmu_graphics import *
import pyaudio as PA
from matplotlib import pyplot as plt
import collections

def generator(start, step):
    s = start
    while True:
        yield s
        s += step


class Module:
    ID = 0
    BorderRadius = 20
    def __init__(self, size=[200,300]):
        #Visual Information
        self.pos = [0,0]
        self.size = size
        self.background = rgb(255, 188, 181)
        self.title = "Default Module"
        self.temporary_background_color = rgb(150,150,150)

        # IO information
        self.sample_rate = 44100
        self.bit_depth = np.int16
        self.sample_size = 256

        # Audio Data 
        self.audio_cache = np.zeros((self.sample_size,))
        self.wave_types =["S", "H", 'N']
        self.id = Module.ID
        Module.ID += 1

        # create knobs as required
        self.knobs = set()

    def _attachKnobs(self):
        pass

    # Visualization
    def Draw(self):
        # background stuff
        # draw a rounded rectangle
        x,y= self.pos
        w,h = self.size
        r = Module.BorderRadius
        # draw an inside rectangle
        drawRect(x+r/2, y+r/2,w-r, h-r, fill=self.background)
        # draw four outside rectangles
        drawRect(x, y+r/2,r/2,h-r,fill=self.background) # left bar
        drawRect(x+w-r/2, y+r/2, r/2,h-r,fill=self.background) # right bar
        drawRect(x+r/2,y,w-r,r/2,fill=self.background) # top bar
        drawRect(x+r/2, y+h-r/2, w-r,r/2,fill=self.background)
        # draw the circles in the corner
        drawCircle(x+r/2, y+r/2, r/2,fill=self.background)
        drawCircle(x+r/2, y+h-r/2, r/2,fill=self.background)
        drawCircle(x+w-r/2, y+h-r/2, r/2,fill=self.background)
        drawCircle(x+w-r/2, y+r/2, r/2,fill=self.background)
        # drawRect(x,y, w,h, fill = self.background)
        drawLabel(self.title, self.pos[0]+self.size[0]/2, self.pos[1]+10, size=16, bold=True)
        # draw all the knobs
        for knob in self.knobs:
            knob.Draw()

    def drawTemporary(self):
        # just draw the label and a light background, before placed
        drawRect(*self.pos, *self.size, fill = self.temporary_background_color, border="black")
        drawLabel(self.title, self.pos[0]+self.size[0]/2, self.pos[1]+10)

    # I/O
    def Input(self, input_data):
        self.audio_cache = input_data

    def Output(self):
        return self.audio_cache
    


class Oscillator(Module):
    def __init__(self, topleft, frequency = 440, amplitude = 1000):
        super().__init__()
        #overrides
        self.pos = topleft
        self.title = "Osc"
        #locals
        self.freq = frequency
        self.amp = amplitude
        self.waveType= self.wave_types[0]
        self.active = False

        # need to store the data which controls the amplitudes and frequencies
        self.amp_control = 0
        self.amp_loc= 0 # place in the amp control we are (if ndarray)
        self.original_amp = self.amp # this is the origional 
        #WLOG freq
        self.freq_control =0
        self.freq_loc = 0
        self.original_freq = self.freq

        self.time = 0

        # Put knobs in our set (from parent)
        self._attachKnobs()

    # Initiation methods
    def _attachKnobs(self):
        # Create an instance of each knob that should be on the oscillator
        x, y = self.pos[0], self.pos[1]
        freqKnob = Knob(
            [x+50,y+100], # position
            "F", # label
            self.setFrequencyStream,# callback to what you want it to change
            False, # it is not an output
            self.getFrequency(), # what the value it should read
            (50,1000), # bounds of values
            2 # increment of turning
        )
        ampKnob = Knob(
            [x+100,y+100],
            "A",
            self.setAmplitudeStream,
            False,
            self.getAmplitude(),
            (0,(2**15-1)/2),
            2
        )
        outputJack = Knob(
            [x+150,y+200],
            "Out",
            self.Output,
            True
        )
        typeJack = Knob(
            [x+50, y+200],
            "Type",
            self.setWaveType,
            False,
            slider=Slider(*self.wave_types)
        )
        self.knobs.add(freqKnob)
        self.knobs.add(ampKnob)
        self.knobs.add(outputJack)
        self.knobs.add(typeJack)

    def Output(self):
        # Generator method for sin wave (other waves to come later)
        return self.getGeneratorOutput()
    
    def getGeneratorOutput(self):
        # use a dictionary intsead of bunch of if statements
        
        wavetypedict ={
            self.wave_types[0] : self.sinWave(),
            self.wave_types[1] : self.squareWave(),
            self.wave_types[2] : self.sawToothWave()
        }
        return wavetypedict[self.waveType]
        

    def sinWave(self):
        #normal sin(x)*a stuff
        self.time += self.sample_size
        return np.array([np.sin(i*(2*np.pi*self.getFrequency())/self.sample_rate)*self.getAmplitude() for i in range(self.time, self.time+self.sample_size)])
        
    def squareWave(self):
        # square is just the sign of the above equation
        return np.array([np.sign(np.sin(i*(2*np.pi*self.getFrequency())/self.sample_rate))*self.getAmplitude() for i in range(self.sample_size)])
    def sawToothWave(self):
        #https://en.wikipedia.org/wiki/Triangle_wave
        pass
    

    #* === Setters ===
    def setAmplitudeStream(self, input_stream):
        # update the input stream for our 
        self.amp_control = input_stream
        self.getAmplitude()
        return self.amp
    
    def setFrequencyStream(self, input_stream):
        self.freq_control = input_stream
        self.getFrequency()
        return self.freq
    
    def setWaveType(self, index):
        # determine what kind of wave we want to outuput
        if 0 <= index < len(self.wave_types):
            self.waveType = self.wave_types[index]
                    
        return self.waveType

    #* === Getters ===
    def getAmplitude(self):
        # WLOG from frequency
        # get what the value would be 
        if not isinstance(self.amp_control, np.ndarray):
            temp = self.amp_control
            if 0<temp<(2**15-1)/2:
                self.amp = temp
                self.original_amp = self.amp # this is the definition of origional amplitude (integer)
        else:
            # get the data we are looking for then check legality
            temp = self.amp_control[self.freq_loc]
            self.amp_loc += 1 # increment index in our control
            self.amp_loc %= len(self.amp_control) # but reset it to 0 when necessary
            if 0<temp+self.original_amp<(2**15-1)/2: # make sure is valid
                self.amp = self.original_amp + temp # don't update the origional, just current
        return self.amp

    def getFrequency(self):
        # WLOG from above
        if not isinstance(self.freq_control, np.ndarray):
            temp = self.freq_control
            if 20<temp<2000:
                self.freq = temp
                self.original_freq = self.freq
        else:
            temp = self.freq_control[self.freq_loc]
            self.freq_loc += 1
            self.freq_loc %= len(self.freq_control)
            if 20<temp+self.original_freq<2000:
                self.freq = self.original_freq * temp
        return self.freq

    def isActive(self):
        return self.active
    
class LFO(Oscillator):
    # the only real difference between and LFO and an oscillator is the 
    def __init__(self, topleft, frequency=10, amplitude=5):
        super().__init__(topleft, frequency, amplitude)
        self.title = "LFO"
        # change the frequency knob settings
        for knob in self.knobs:
            if knob.label == "F":
                knob.increment = 0.1    
    
    def setFrequencyStream(self, amount):
        if not isinstance(amount, (np.int64, int, float)):
            amount= next(amount)
        if 0 < amount < 20: self.freq = amount
        return self.freq

class Mixer(Module):
    def __init__(self, position):
        super().__init__()
        self.title = "Mixer"
        self.pos= position
        # Locals
        self.isFull= False
        self._attachKnob()

        
        # Audio Output Information (Can't use until MVP)
        self.stream = PA.PyAudio().open(
            rate = self.sample_rate,
            channels = 1,
            format = PA.paInt16,
            output=True,
            frames_per_buffer=self.sample_size
        )
        
        self.isActive = True

    def startThread(self):
        self.thread.start()

    def hasNonZeroData(self):
        for knob in self.knobs:
            return not knob.isEmpty

    def _attachKnob(self):
        x,y = self.pos[0], self.pos[1]
        input_knob = Knob(
            [x+50, y+100],
            "In",
            self.Input,
            False
        )
        self.knobs.add(input_knob)

    def Output(self):
        
        return
    
    def playSound(self, connection_manager):
        # Make this always running until turned off
        while self.isActive:
            connection_manager.updateConnections()
            # make sure our audio cache is the correct type
            data = self.getAudioCache()
                 # output it to our stream 
            self.stream.write(data.tobytes())

    def outputPLT(self):
        xs= np.linspace(0,1, len(self.audio_cache))
        plt.plot(xs, self.audio_cache)
        plt.show()

    def getAudioCache(self):
        temp = np.array(self.audio_cache).astype(np.int16)
        self.audio_cache = np.zeros(len(temp))
        return temp

    def Input(self, input_data):
        self.audio_cache = input_data

    def close(self):
        self.isActive=False
        self.stream.close()

        
    

class Adder(Module):
    def __init__(self, position):
        super().__init__()
        self.pos = position
        self.title = "Adder"

        self._attachKnobs()
        
        self.A_cache = None
        self.B_cache = None

    def _attachKnobs(self):
        x,y = self.pos[0], self.pos[1]
        knobA = Knob(
            [x+50,y+100],
            "A",
            self.In1,
            False,
            0,
            (0,0),
            0
        )
        knobB = Knob(
            [x+100,y+100],
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

    def In1(self, input):
        assert(len(input) == self.sample_size)
        self.A_cache = input
    def In2(self, input):
        assert(len(input) == self.sample_size)
        self.B_cache = input

    def Output(self):
        return self._limitedValues()

    def _limitedValues(self):
        if self.B_cache is not None and self.A_cache is not None:
            res = self.B_cache + self.A_cache
            max_val = 2**15-1/2
            np.clip(res, -max_val, max_val, out=res)
            return res
        return np.zeros(self.sample_size)


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
        # since we're outputting something larger than the sample size, keep track of where we are in the envelope
        self.output_queue= collections.deque()

        # now for the streams and locs of each

        self.displayEnvelope = False
        
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
        x,y = self.pos[0], self.pos[1]
        attKnob = Knob(
            [x+40, y+100],
            "Attack",
            self.setAttack,
            False,
            0.1,
            (0,2),
            0.1
        )
        decayKnob = Knob(
            [x+80, y+100],
            "Decay",
            self.setDecay,
            False,
            0,
            (0,2),
            0.1
        )
        sustainKnob = Knob(
            [x+120, y+100],
            "Sustain",
            self.setSustain,
            False,
            1,
            (0,1),
            0.1
        )
        releaseKnob = Knob(
            [x+160, y+100],
            "Release",
            self.setRelease,
            False,
            0,
            (0,2),
            0.1
        )
        outputKnob = Knob(
            [x+160,y+200],
            "Out",
            self.Output,
            True
        )
        inKnob = Knob(
            [x + 40, y + 200],
            "In",
            self.Input,
            False
        )
        self.knobs.add(attKnob)
        self.knobs.add(decayKnob)
        self.knobs.add(sustainKnob)
        self.knobs.add(releaseKnob)
        self.knobs.add(outputKnob)
        self.knobs.add(inKnob)
    
    # want to fill my audio packet up to the number of samples
    # for me to perform the 
    def Input(self, input):
        #rotate array and put input on the front
        self.audio_cache = np.roll(self.audio_cache, len(input))
        self.audio_cache[:len(input)] = input
        
    def Output(self):
        # we can create a numpy array of samples from 0-1 and then
        # multiply the envelope
        if self.displayEnvelope:
            xs = np.linspace(0,len(self.envelope), len(self.envelope))
            plt.plot(xs, self.envelope, 'r-')
            plt.plot(xs, self.audio_cache, 'b-')
            plt.show()
        self._applyFilter()
        return [self.output_queue.pop() for i in range(self.sample_size)]


    def _applyFilter(self)->np.ndarray:
        final_signal = self.audio_cache * self.envelope
        # normalize it to approrpriate means
        if np.max(final_signal) == 0: return final_signal
        final_signal /= np.max(np.abs(final_signal))
        for i in final_signal:
            self.output_queue.appendleft(i)
    

    #* Setters
    def setAttack(self, amount):
        if not isinstance(amount, (np.int64, int, float)):
            amount = next(amount)
        self.attack = amount
        return self.attack
    
    def setDecay(self, amount):
        if not isinstance(amount, (np.int64, int, float)):
            amount = next(amount)
        self.decay = amount
        return self.decay
    
    def setSustain(self, amount):
        if not isinstance(amount, (np.int64, int, float)):
            amount = next(amount)
        self.sustain = amount
        return self.sustain
    def setRelease(self, amount):
        if not isinstance(amount, (np.int64, int, float)):
            amount = next(amount)
        self.release = amount
        return self.release
    
class LPF(Module):
    def __init__(self, pos):
        super().__init__()
        self.pos = pos
        self.title = "LPF"
        self.freq_control = np.zeros(self.sample_size)
        # Note: audio cache is bigger than 1 sample size because better for fft accuracy
        self.audio_cache = np.zeros(self.sample_size*2)

        self.bandwidth = 100

        self.freq_loc = 0
        self.freq = 450
        self.original_freq = 0

        self._attachKnobs()

    def _attachKnobs(self):
        x,y=self.pos[0], self.pos[1]
        freqKnob = Knob(
            [x+50,y+100],
            "F",
            self.setFrequencyStream,
            False,
            self.getFrequency(),
            (50,2000),
            2
        )
        outputJack = Knob(
            [x+150,y+200],
            "Out",
            self.Output,
            True
        )
        input_knob = Knob(
            [x+50, y+200],
            "In",
            self.Input,
            False
        )
        self.knobs.add(input_knob)
        self.knobs.add(freqKnob)
        self.knobs.add(outputJack)

    def setFrequencyStream(self, input_stream):
        self.freq_control = input_stream
        self.getFrequency()
        return self.freq
    
    def Input(self, data):
        self.audio_cache = np.roll(self.audio_cache, len(data))
        self.audio_cache[:len(data)] = data
        

    def Output(self):
        self.updateCache()
        super().Output()
    
    def getFrequency(self):
        # WLOG from above
        if not isinstance(self.freq_control, np.ndarray):
            temp = self.freq_control
            if 20<temp<2000:
                self.freq = temp
                self.original_freq = self.freq
        else:
            temp = self.freq_control[self.freq_loc]
            self.freq_loc += 1
            self.freq_loc %= len(self.freq_control)
            if 20<temp+self.original_freq<2000:
                self.freq = self.original_freq + temp
        return self.freq

    def fftPlot(self, sig, dt=1/44100, plot=True):
        # https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
        # Here it's assumes analytic signal (real signal...) - so only half of the axis is required

        if dt is None:
            dt = 1
            t = np.arange(0, sig.shape[-1])
            xLabel = 'samples'
        else:
            t = np.arange(0, sig.shape[-1]) * dt
            xLabel = 'freq [Hz]'

        if sig.shape[0] % 2 != 0:
            t = t[0:-1]
            sig = sig[0:-1]

        sigFFT = np.fft.fft(sig) / t.shape[0]  # Divided by size t for coherent magnitude

        freq = np.fft.fftfreq(t.shape[0], d=dt)

        # Plot analytic signal - right half of frequence axis needed only...
        firstNegInd = np.argmax(freq < 0)
        freqAxisPos = freq[0:firstNegInd]
        sigFFTPos = 2 * sigFFT[0:firstNegInd]  # 2 because of magnitude of analytic signal

        if plot:
            plt.figure()
            plt.plot(freqAxisPos, np.abs(sigFFTPos))
            plt.xlabel(xLabel)
            plt.ylabel('mag')
            plt.xscale('log')
            plt.title('Analytic FFT plot')
            plt.show()

        return sigFFTPos, freqAxisPos


    def updateCache(self):
        
        # self.fftPlot(self.audio_cache)
        self.updateAudioCache()
        # self.fftPlot(self.audio_cache)
        

    def updateAudioCache(self):
        #https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
        f_c = self.getFrequency() / self.sample_rate # frequency cutoff
        b = self.bandwidth / self.sample_rate # the width of the band
        N = int(np.ceil(4/b))
        if N%2 != 0: N += 1 # we want N to be odd
        xs = np.arange(N)
        # the sinc filter itslef
        sinc = np.sinc(2*f_c*(xs-(N-1)/2))
        # blackman window
        blackman = np.blackman(N)
        # the product of the two
        s_b_product = sinc * blackman
        # unity gain ( normalize )
        unity_gain = s_b_product / np.sum(s_b_product) 
        
        #* apply the unity gain to the list as required
        self.audio_cache = np.convolve(self.audio_cache, unity_gain)
        # print(self.audio_cache)


    
    
    

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
                slider = None):
        # Visual Information
        self.pos=position
        self.jack_size = 10 # diameter of the hole 
        self.knob_size = 15 # diameter of the turny bit
        self.color = rgb(67, 170, 139)
        self.label = label
        self.slider_length = 20

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
            drawCircle(*self.pos, self.jack_size, fill = 'black')
            # draw the value and label
            drawLabel(self.label, self.pos[0], self.pos[1]+self.jack_size+10)
            drawLabel(str(self.value), self.pos[0], self.pos[1]-self.jack_size-10)

    def turnLeft(self):
        if self.isOutput(): return
        self.value -= self.increment
        if self.slider != None:
            self.slider.moveLeft()
        self.setValue()

    def turnRight(self):
        if self.isOutput(): return
        self.value += self.increment
        if self.slider != None:
            self.slider.moveRight()
        self.setValue()


    def isOutput(self):
        return self.output

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
        for connector in self.connections:
            connector.executeConnection()

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
