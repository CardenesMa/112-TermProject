# Eurorack-112
Eurorack-112 is a digital modular synthesizer game that you can play with as a character in a platformer game. There's an infinite number of unique sounds you can create, so have fun exploring the world of modular synthesis. 

## Installation for Mac
Run the following commands in the terminal: 

`brew install portaudio`

`pip3 install -r requirements.txt`

This will install the packages required for this project.
This assumes that pip, python, and brew are already installed on your device.  

Run `main.py` to see the project in action. Do not alter or move the other files outside the relative directory.


# Modules
How to use the modules: 
By attaching modules together you can create a (hypothetically) infinite number of unique sounds. Each module has a unique purpose, all of which listed below for your reference. There are two main kinds of modules: generators and controllers. Generators create a mathematically pure sound wave, or 'ingredients' for making a sound wave. Controllers take a sound wave and change them in some way. At the end of any chain of modules is the mixer, which is where we hear the sounds come out. Any chain should look like this: `Generator--> (in) Controller(s) --> (in) Mixer`. Note that any arrow implies that the plug goes explicitly from the `out` plug to a specified plug. All colored plugs are inputs*. Here are some cool chains that you can try out:

* `Randomizer --> (F) Oscillator --> (in) Mixer` 

* `LFO --> (A) Oscillator --> (in) Filter --> (in) Mixer`

* `Oscillator (2x) --> (A,B) Adder --> (in) Mixer`

```
Randomizer --> (F) Oscillator --\
                                (A)  (B)  Adder --> Mixer
Randomizer --> (F) Oscillator  ----- /
```


\* Some cannot take stream inputs as of yet.
## Generators

### Osc (Oscillator)
The oscillator takes a frequency, an amplitude, and a choice of waveforms to create a sound wave. The  `frequency` (hz) and `amplitude` can be automated; most commonly with an LPF. The waveform choices are `sine (S), square (H), and sawtooth (N)`. This is the most foundational of the modules. 

### Randomizer
The randomizer module outputs a random frequency relative to A4. The `bottom` and `top` ends of the range tell you how many semitones away from A4 the randomizer can chose from. The `rate` option tells you the hertz for which it will choose a random value in that range. The randomizer outputs an integer equal-tempered frequency. The main usage for the randomizer is to connect it to the frequency input on the oscillator for producing the frequencies generated by the randomizer. 

### LFO (Low-Frequency Oscillator)
An LFO is very similar to an oscillator, and it works the same way. The main distinction is that it works at a significantly lower frequency. The main use for an LFO is to automate the inputs of modules. This can be done by connecting the output of the LFO to turnable knobs such as the frequency on an oscillator or the cutoff on a LPF.

## Controllers

### LPF (Low-Pass Filter)
A low-pass filter is a method of removing frequencies above a certain range. The `cutoff` is the location at which the filter removes frequencies explicitly higher than it. It changes the shape of sawtooth and square waves to remove their harsher overtones. The LPF can be used anywhere in the module chain.

### Filter (Envelope Filter)
The envelope filter takes a signal and shapes it's amplitude. Contrary to the LPF, it is explicitly an amplitude filter. The `attack` represents the time to go from 0 to maximum amplitude, `decay` controls how quickly it drops to the `sustain` level of amplitude, and `release` determines how long before the amplitude trails off to 0. The most common use for the filter is directly after an oscillator. 

## Adder 
The adder module does exactly what it sounds like it does. It takes two signals, `a` and `b`, and adds them together, normalizing the frequency to the maximum allowed at the bit depth of the project.


## Note: The openMixerData and s.txt are for debugging and visualizing the waveform in the mixer. If something goes wrong, let me know and use that to post a picture of the waveform! 