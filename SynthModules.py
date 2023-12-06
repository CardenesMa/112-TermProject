# keeping track of things like rate and delay
import time
# all arrays should be numpy for compatability sake
import numpy as np
from cmu_graphics import drawRect, drawLabel, drawCircle, rgb
# For plating all the audio
import pyaudio as PA
# Ranxomizer requires this
import random
# Used to make the low pass filter. 
from scipy.signal import butter, filtfilt
# local
from ModuleOverlays import Knob, Slider

class Module:
    """Parent of all modules. Contains the master attributes such as size, title, 
    and sets the idenfitication of a module. Basic functions that don't require overrides are 
    listed here.

    Returns:
        np.ndarray: array of length <sample_size> which is the output of the internal algorithm
    """
    ID = 0
    BorderRadius = 20
    def __init__(self, size=[200,300]):
        #Visual Information
        self.pos = [0,0]
        self.size = size
        self.background = rgb(255, 188, 181)
        self.title = "Default Module"
        self.temporary_background_color = rgb(150,150,150)

        # information for y values of knobs relative to the position
        self.knoby_top = 100
        self.knoby_bottom = 200

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
        """Draws a basic rounded rectangle along with each knob in the self.knobs set. 
        Only to be called in the View of an MVC structure.
        """
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
        """When choosing what module to place, this draws a preview of what you will place.
        """
        # just draw the label and a light background, before placed
        drawRect(*self.pos, *self.size, fill = self.temporary_background_color, border="black")
        drawLabel(self.title, self.pos[0]+self.size[0]/2, self.pos[1]+10)

    # I/O
    def Input(self, input_data):
        # default to saving the input data internally
        self.audio_cache = input_data

    def Output(self):
        # default to outputting whatever is stored
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
        # setting the wave type to sin
        self.waveType= self.wave_types[0]
        # setting a custom background color
        self.background = rgb(190, 181, 255)
        # need to store the data which controls the amplitudes and frequencies
        self.amp_control = 0
        self.amp_loc= 0 # place in the amp control we are (if ndarray)
        self.original_amp = self.amp # this is the original 
        #WLOG freq
        self.freq_control =0
        self.freq_loc = 0
        self.original_freq = self.freq

        # used for indexing when making the wave signal
        self.time = 0
        # Put knobs in our set (from parent)
        self._attachKnobs()

    # Initiation methods
    def _attachKnobs(self):
        # Create an instance of each knob that should be on the oscillator
        x, y = self.pos[0], self.pos[1]
        freqKnob = Knob(
            [x+50,y+self.knoby_top], # position
            "F", # label
            self.setFrequencyStream,# callback to what you want it to change
            False, # it is not an output
            self.getFrequency(), # what the value it should read
            (50,1000), # bounds of values
            2 # increment of turning
        )
        ampKnob = Knob(
            [x+100,y+self.knoby_top],
            "A",
            self.setAmplitudeStream,
            False,
            self.getAmplitude(),
            (0,(2**15-1)/2),
            2
        )
        outputJack = Knob(
            [x+150,y+self.knoby_bottom],
            "Out",
            self.Output,
            True
        )
        typeJack = Knob(
            [x+50, y+self.knoby_bottom],
            "Type",
            self.setWaveType,
            False,
            slider=Slider(*self.wave_types)
        )
        # add in-place 
        self.knobs.add(freqKnob)
        self.knobs.add(ampKnob)
        self.knobs.add(outputJack)
        self.knobs.add(typeJack)

    def Output(self):
        # Get the approprite awave for the appropriate job
        if self.waveType == self.wave_types[0]:
            return self.sinWave()
        elif self.waveType == self.wave_types[1]:
            return self.squareWave()
        elif self.waveType == self.wave_types[2]:
            return self.sawToothWave()
        else:
            return np.zeros(self.sample_size)

    def sinWave(self):
        #normal sin(x)*a stuff
        self.time += self.sample_size
        self.time %= self.sample_rate
        return np.array([np.sin(i*((2*np.pi*self.freq)/self.sample_rate))*self.amp for i in range(self.time, self.time + self.sample_size)])
        
    def squareWave(self):
        # square is just the sign of the above equation
        self.time += self.sample_size
        self.time %= self.sample_rate
        return np.array([np.sign(np.sin(i*(2*np.pi*self.getFrequency())/self.sample_rate))*self.getAmplitude() for i in range(self.time, self.time+ self.sample_size)])
    def sawToothWave(self):
        #https://en.wikipedia.org/wiki/Sawtooth_wave
        # think of it as linear from 0-amp where we just say it's t+amp/sample_size
        self.time += self.sample_size
        self.time %= self.sample_rate
        return np.array([t*self.getAmplitude()/self.sample_size for t in range(self.sample_size)])
        
    

    #* === Setters ===
    def setAmplitudeStream(self, input_stream):
        # update the input stream for our 
        self.amp_control = input_stream
        self.getAmplitude()
        return self.amp
    
    def setFrequencyStream(self, input_stream):
        # WLOG as above
        self.freq_control = input_stream
        self.getFrequency()
        return self.freq
    
    def setWaveType(self, index):
        # determine what kind of wave we want to output
        if 0 <= index < len(self.wave_types):
            self.waveType = self.wave_types[index]
        # we return the index not the wave type-- important for sliders
        return index

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
            return self.freq
        else:
            temp = self.freq_control[self.freq_loc]
            if 20<temp+self.original_freq<2000:
                self.freq = self.original_freq + temp
            self.freq_loc += 1
            self.freq_loc %= len(self.freq_control)
            return self.freq
    
class LFO(Oscillator):
    # the only real difference between and LFO and an oscillator is the default amplitudes and freqs
    def __init__(self, topleft, frequency=20, amplitude=5):
        super().__init__(topleft, frequency, amplitude)
        self.title = "LFO"
        # change the frequency knob settings instead of re-adding them
        for knob in self.knobs:
            if knob.label == "F":
                knob.increment = 0.1    
            if knob.label == "A":
                knob.increment = 0.5
    
    def setFrequencyStream(self, input_stream):
        # see Oscillator 
        self.freq_control = input_stream
        self.getFrequency()
        return self.freq
    
    def getFrequency(self):
        # WLOG from above
        if not isinstance(self.freq_control, np.ndarray):
            temp = self.freq_control
            if 0<temp<20:
                self.freq = temp
                self.original_freq = self.freq
            return self.freq
        else:
            temp = self.freq_control[self.freq_loc]
            if 0<temp+self.original_freq<20:
                self.freq = self.original_freq + temp
            self.freq_loc += 1
            self.freq_loc %= len(self.freq_control)
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
        # used to control the audio stream
        self.isActive = True
        # used for debugging wen we see 
        self.temp_data = np.zeros(self.sample_size*100)
        self.itern =0


    def hasNonZeroData(self):
        for knob in self.knobs:
            return not knob.isEmpty

    def _attachKnob(self):
        x,y = self.pos[0], self.pos[1]
        input_knob = Knob(
            [x+50, y+self.knoby_top],
            "In",
            self.Input,
            False,
            0,
            (0,0),
            0,
            isInput= True
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

    def getAudioCache(self):
        temp = np.array(self.audio_cache).astype(np.int16)
        self.audio_cache = np.zeros(len(temp))
        return temp

    def Input(self, input_data):
        if not isinstance(input_data, np.ndarray):
            return 0
        self.audio_cache = input_data

        # In case we want to look at some of this data, we save it and open it in some other application.
        self.itern += 1
        if self.itern == 100:
            self.temp_data.tofile("./s.txt")
            self.itern =0
            # print("Data Stored")
        # else:
        #     self.temp_data = np.roll(self.temp_data, len(input_data))
        #     self.temp_data[:len(input_data)] = input_data


    def close(self):
        self.isActive=False
        self.stream.close()

        
    

class Adder(Module):
    def __init__(self, position):
        super().__init__()
        self.pos = position
        self.title = "Adder"
        self.background = rgb(255, 181, 218)
        self._attachKnobs()
        
        self.A_cache = None
        self.B_cache = None

    def _attachKnobs(self):
        x,y = self.pos[0], self.pos[1]
        knobA = Knob(
            [x+50,y+self.knoby_top],
            "A",
            self.In1,
            False,
            0,
            (0,0),
            0,
            isInput=True
        )
        knobB = Knob(
            [x+100,y+self.knoby_top],
            "B",
            self.In2,
            False,
            0,
            (0,0),
            0,
            isInput=True
        )

        outKnob = Knob(
            [x+160, y+self.knoby_bottom],
            "Out",
            self.Output,
            True
        )
        
        self.knobs.add(knobA)
        self.knobs.add(knobB)
        self.knobs.add(outKnob)

    def In1(self, input):
        if not isinstance(input, np.ndarray):
            return 
        self.A_cache = input
    def In2(self, input):
        if not isinstance(input, np.ndarray):
            return
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
                 sustain = 0.5, release = 1):
        super().__init__()
        self.pos = position
        self.title = "Filter"
        self.background = rgb(130, 171, 254)

        # ASDR
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release= release
        
        # since we're outputting something larger than the sample size, keep track of where we are in the envelope
        self._setSamples()
                
        self._attachKnobs()
        self._createEnvelope()

    def _setSamples(self):
        # This is used to update the samples and size of the filter when needed
        self.duration = int(self.attack+self.release+self.decay)
        # the number of samples which holds the asdr as needed
        self.attack_samples = int(self.sample_rate * self.attack)
        self.decay_samples = int(self.sample_rate * self.decay)
        self.sustain_samples = int((self.duration-self.decay-self.release-self.attack)*self.sample_rate)
        self.sustain_amount = self.sustain
        self.release_samples = int(self.sample_rate * self.release)
        # total number of samples in the filter
        self.num_samples = self.duration*self.sample_rate
        # is the values of the envelope itself at that length 
        self.envelope = np.zeros(self.num_samples)
        # set the audio cache to the appropriate length
        self.audio_cache = np.zeros(self.num_samples)
        self.output_samples = np.zeros(self.sample_size)




    def _createEnvelope(self):
        self._setSamples()
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
        self.envelope[Ea+Ed:Ea+Ed+Es] = self.sustain
        # release
        self.envelope[-Er:]=np.linspace(self.sustain_amount,0,Er)
        
        


    def _attachKnobs(self):
        x,y = self.pos[0], self.pos[1]
        attKnob = Knob(
            [x+40, y+self.knoby_top],
            "Attack",
            self.setAttack,
            False,
            0.1,
            (0,2),
            0.1
        )
        decayKnob = Knob(
            [x+80, y+self.knoby_top],
            "Decay",
            self.setDecay,
            False,
            0,
            (0,2),
            0.1
        )
        sustainKnob = Knob(
            [x+120, y+self.knoby_top],
            "Sustain",
            self.setSustain,
            False,
            1,
            (0,1),
            0.1
        )
        releaseKnob = Knob(
            [x+160, y+self.knoby_top],
            "Release",
            self.setRelease,
            False,
            0,
            (0,2),
            0.1
        )
        outputKnob = Knob(
            [x+160,y+self.knoby_bottom],
            "Out",
            self.Output,
            True
        )
        inKnob = Knob(
            [x + 40, y + self.knoby_bottom],
            "In",
            self.Input,
            False,
            isInput= True
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
        self.audio_cache =np.roll(self.audio_cache, len(input))
        self.audio_cache[:len(input)] = input
        
    def Output(self):
        # we can create a numpy array of samples from 0-1 and then
        # multiply the envelope
        self._applyFilter()
        return self.output_samples

    def _applyFilter(self):
        if self.envelope.size != self.audio_cache.size: 
            pass
        else:
            self.output_samples = self.envelope*self.audio_cache
    

    #* Setters
    #TODO: Update to handle inputs like the other methods
    def setAttack(self, amount):
        if not isinstance(amount, (np.int64, int, float)):
            amount = next(amount)
        self.attack = amount
        self._createEnvelope()
        return self.attack
    
    def setDecay(self, amount):
        if not isinstance(amount, (np.int64, int, float)):
            amount = next(amount)
        self.decay = amount
        self._createEnvelope()
        return self.decay
    
    def setSustain(self, amount):
        if not isinstance(amount, (np.int64, int, float)):
            amount = next(amount)
        self.sustain = amount
        self._createEnvelope()
        return self.sustain
    
    def setRelease(self, amount):
        if not isinstance(amount, (np.int64, int, float)):
            amount = next(amount)
        self.release = amount
        self._createEnvelope()
        return self.release

    
class LPF(Module):
    def __init__(self, pos):
        super().__init__()
        self.pos = pos
        self.title = "LPF"
        self.freq_control = np.zeros(self.sample_size)
        # Note: audio cache is bigger than 1 sample size because better for fft accuracy
        self.audio_cache = np.zeros(self.sample_size)
        self.background = rgb(255, 181, 181)

        # same stuff as above in oscillator input jack
        self.freq_loc = 0
        self.freq = 450
        self.original_freq = 0

        # must call this so that we have somewhere to store out coefficients
        self.updateCoefficients()

        self._attachKnobs()

    def _attachKnobs(self):
        x,y=self.pos[0], self.pos[1]
        freqKnob = Knob(
            [x+50,y+self.knoby_top],
            "F",
            self.setFrequencyStream,
            False,
            self.getFrequency(),
            (50,2000),
            2
        )
        outputJack = Knob(
            [x+150,y+self.knoby_bottom],
            "Out",
            self.Output,
            True
        )
        input_knob = Knob(
            [x+50, y+self.knoby_bottom],
            "In",
            self.Input,
            False,
            isInput= True
        )

        self.knobs.add(input_knob)
        self.knobs.add(freqKnob)
        self.knobs.add(outputJack)

    def setFrequencyStream(self, input_stream):
        self.freq_control = input_stream
        self.getFrequency()
        self.updateCoefficients()
        return self.freq
    
    def updateCoefficients(self):
        normal_cutoff = self.freq / (0.5*self.sample_rate) # nyquist! 
        # we're keeping the order of the butter at 2 but may be nice to say 3
        self.coef_a,self.coef_b = butter(3, normal_cutoff, 'low',False ) # get the coefficients from butter
    
    
    def Input(self, data):
        # move the data over then add the new stuff
        self.audio_cache = data
        

    def Output(self):
        self.updateCache()
        return self.audio_cache
    
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
    
    def updateCache(self):
        self.audio_cache = filtfilt(self.coef_a, self.coef_b,self.audio_cache) # apply filter both ways


class Randomizer(Module):
    def __init__(self, pos, size=[200, 300], note_range = (-8,8), rate=1):
        super().__init__(size)
        self.title="Randomizer"
        # Create random outputs 
        # going to use equal temperment for ease's sake
        self.tr2 = 2**(1/12) # twelfth root of two
        self.range = note_range # this is the range of notes as far from A as it gets
        self.rate = rate # since rate is in notes/second, we can get the 
        # we default to 440
        self.freq = 440 
        self.background = rgb(157, 251, 164)
        self.pos = pos
        self.last_time = 0 # keeping track of our time for outputs
        self._attachKnobs()

    def Input(self, input_data):
        # for now, this can't take any input
        return 0
    
    def Output(self):
        # and this only outputs integers for the sake of simplicity
        self._updateFreq()
        return self.freq


    def _updateFreq(self):
        t1 = time.time()
        if self.rate == 0: return
        if t1 - self.last_time >= 1/self.rate: # we want to be explicit in our timing
            # change the note value once we've reached our increment within reasonable bounds
            self.last_time = t1
            freq_choice = random.randint(self.range[0], self.range[1])
            self.freq = self._getFrequencyFromNote(freq_choice)
        return self.freq
        
        

    def _attachKnobs(self):
        super()._attachKnobs()
        x,y=self.pos
        bottomKnob = Knob(
            [x+130,y+self.knoby_top],
            "Botom",
            self.setBottom,
            False,
            -8,
            (-30,30),
            1
        )
        topKnob = Knob(
            [x+80, y+self.knoby_top],
            "Top",
            self.setTop,
            False,
            8,
            (-29,31),
            1
        )
        rateKnob = Knob(
            [x+30, y+self.knoby_top],
            "Rate Hz",
            self.setRate,
            False,
            1,
            (0,5),
            0.1
        )
        outputJack = Knob(
            [x+150,y+self.knoby_bottom],
            "Out",
            self.Output,
            True
        )
        self.knobs.add(bottomKnob)
        self.knobs.add(topKnob)
        self.knobs.add(rateKnob)
        self.knobs.add(outputJack)


    def _getFrequencyFromNote(self, note):
        # using a as relative
        return int(self.tr2**note*440)
    
    def setBottom(self, stream):
        if not isinstance(stream, np.ndarray): # onlt want to do integers for now
            _,t = self.range
            if stream < t:
                self.range = (stream, t) # set the new bottom to the input
        return self.range[0]
    
    def setTop(self, stream):
        if not isinstance(stream, np.ndarray): #WLOG bottom
            b, _ = self.range
            if stream > b:
                self.range = (b,stream)

        return self.range[1]
    
    def setRate(self, stream):
        if not isinstance(stream ,np.ndarray):
            if 0 <= stream <= 5:
                self.rate = stream
        return self.rate

    
class Sequencer(Randomizer): 
    # The only real difference between a sequencer and a randomizer is the 
    # Algorithm for generating the notes. Inherit for simplicity's sake
    def __init__(self, pos, size=[200, 300], note_range=(-8, 8), rate=1, direction=1):
        super().__init__(pos, size, note_range, rate)
        self.title= "Sequencer"
        # Direction :
        # -1 : down, 0: up and down, 1: up
        self.direction = direction
        self.direction_momentum = 1 # this is for the up and down -- keeping track
        self.note_value = 0
        self.background = rgb(253, 174, 174)
        # similar as goes with frequency in oscillator
        self.step = 1
        self.step_control = np.zeros(self.sample_size)
        self.original_step= self.step
        self.step_loc = 0


    def _attachKnobs(self):
        # want to add one more knob! 
        super()._attachKnobs() 
        directionSlider = Slider("U","U+D", "D")
        x,y = self.pos
        dKnob = Knob(
            [x+50, y+self.knoby_bottom],
            "",
            self.setDirection,
            slider = directionSlider
        )
        stepknob = Knob(
            [x+180, y+self.knoby_top],
            "Step",
            self.setStepStream,
            False,
            1,
            (1,self.range[1]-self.range[0]+1),
            1
        )
        self.knobs.add(dKnob)
        self.knobs.add(stepknob)

    def setDirection(self, index):
        # determine what kind of wave we want to output
        if index == 0:
            self.direction = 1
        elif index == 1:
            self.direction = -1
        elif index == 2:
            self.direction = 0
        return index
    
    def setStepStream(self, input_stream):
        # see Oscillator 
        self.step_control = input_stream
        self.getStep()
        return self.step
    
    def getStep(self):
        # WLOG from above
        if not isinstance(self.step_control, np.ndarray):
            temp = self.step_control
            if 0<temp<20:
                self.step = temp
                self.original_step = self.step
            return self.step
        else:
            temp = self.step_control[self.step_loc]
            if 0<temp+self.original_step<20:
                self.step = self.original_step + temp
            self.step_loc += 1
            self.step_loc %= len(self.step_control)
            return self.step


    def _updateFreq(self):
        t1 = time.time()
        if self.rate == 0: return
        if t1 - self.last_time >= 1/self.rate: # we want to be explicit in our timing
            # change the note value once we've reached our increment within reasonable bounds
            self.last_time = t1
            if self.direction == 1: # if we're going up,
                self.note_value += int(self.step)

            elif self.direction == -1: # down
                self.note_value -= int(self.step)

            elif self.direction == 0: # up and down
                if self.note_value >= self.range[1]:
                    self.direction_momentum = -1
                elif self.note_value <= self.range[0]:
                    self.direction_momentum = 1
                # once we get the momentum (up or down) we apply it to step and go that way
                self.note_value += self.direction_momentum * int(self.step)
                
                
            else:
                print("invalid direction")
            # Wrap around the frequency (modulo more complitated in this case)
            if self.direction == 1 or self.direction == -1:
                if self.note_value > self.range[1]:
                    self.note_value = self.range[0]
                if self.note_value < self.range[0]:
                    self.note_value = self.range[1]
            
            
            # return the outputting frequency
            self.freq = self._getFrequencyFromNote(self.note_value)
        return self.freq  

