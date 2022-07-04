# LEDMSI v0.1 Design notes: Field test results
## Summary
This document summarises what we learned from the first field trials of the LEDMSI system in June 2022 at Ginnerup and Sorte Muld.

(issues #1, #2, #3 & #4)

## Easy fixes
### The system is too slow
**The problem:** It takes nearly 5 minutes to acquire the full image stack. This is too long.  

**Solutions:**  

1. The code surrently intitialises the arduino that controls the LEDS for every wavelength. It does this via `PyFirmata`, which takes a few seconds to intialise the board. Fix this so that the arduino is intialised in the `__init__` method of the `CameraControl`class.  
2.  It takes a few seconds to write each image to file. Part of this may be due to [a known issue with the raspberry pi](https://forums.raspberrypi.com/viewtopic.php?t=245931) and external storage. This issue has a known fix. Further to applying this fix we'll test writing to RAM via `initramfs` and then copying this to disk thereafter  
3. Currently we're using `libcamera`to handle the camera. The slow write speed may further be mitigated by better configuring `libcamera`. This includes increasing the size of the buffer and squashing output  
4. The way we're calling `libcamera` from the python code requires initialising the camera every time we take a photo. `picamera2` does not require this so we will test the code with `picamera2`, with the caveat that this library is an alpha release and may be subject to API changes which is why it was avoided in the first place.
5. Increase gain to reduce exposure times.

***Cost:*** 1-2 Days coding
### Not enough user feedback 
**The problem:** There's not enough feedback provided to the end user on progress / status. This is both an audio and visual problem- when the camera is covered by lightproof material it's difficult for the user to know that it's finished.  

(Issue #5)

**Solutions:**
  
1. Enable the preview window in `libcamera`. Currently we're calling it with no preview. We can also write information on the current LED wavelength to the preview window  
2. Add a speaker / buzzer to provide audio feedback when the system is covered.

***Cost:*** 0.5 days. Buzzer DKK 20. 

### Poor thermal management

**The problem:** At Sorte Muld enough heat was generated that the system went into thermal shutdown. Furthermore the 3D printed material of the case shrank, meaning that the touch screen no longer fitted in its pocket. This is likely a problem with using the system facing down, as all heat is trapped by this part and was likely exacerbated by the system being covered by black plastic on a sunny day. 

**Solutions:**

1. Redesign this part with more clearance in the pocket for the screen. Add some ventilation around the edges. 
2. Add a cooling fan for the Raspberry Pi to increase air circulation as the Pi is likely the greatest generator of heat.
3. Try to find lightproof material that's reflective on one side, or add a reflective layer to it (survival blanket?)

***Cost:*** 0.5 days. Fan DKK 50.

### Masking ambient light 
**The problem:** Minor light leaks are caused by interface between the frame holding the lightproof fabric and uneven surfaces. 

**Solutions:**

1. Find soft foam material for this interface. Pipe lagging is too rigid to conform- so softer than this.

***Cost:*** Low- requires a trip to a fabric shop to see what they have.

### Supporting the camera during exposure ###
**The problem:** The exposures are too long to hand-hold the camera. However, using a conventional tripod is a problem in the confined space of the trench, and in deep trenches inverting the tripod column halfway is time consuming. It also risks movement causing blur / misregistration between exposures.

**Solutions**

1. Build a linear stage approximately 1m high to hold and move the system parallel to the surface. Twin aluminium rails actuated by a screw? Can be supported by a weighted foot at the base or suspended from the top of the trench.

***Cost:*** 2 days. 200 DKK for aluminium


## Requiring Redesign ##

### Poor fluorescence performance
**The problem:** Curently most of the fluorescence contribution to the images is overwhelmed by reflectance. We need to better discretise the flourescence contribution to luminance. Currently we're attempting to identify this by using the offset between hue angles- which may be enough to determine which filters are useful.

**Solutions:**

1. Add a filter wheel for the camera. This will require redesign and remanufacture of the camera spacer so is a substantial undertaking, and the filters are expensive.
2. Add further cameras with individual filters. Requires a multiplexer for the camera connector. Even more expensive

**Notes:**
I've ordered 4 filters- 400nm, 420nm, 470nm and 510nm

***Cost:*** 3 days. 600 DKK for filters. Plus cost of small stepper motor *or* 1000 DKK for four cameras, plus 600 DKK for filters plus multiplexer board for cameras

## Things we can't do much about
### Creatures
**The problem:** At Sorte Muld numerous insects and invertabrates were observed in the images. This is a problem as they move, causing misrepresentative results comparing bands. 

**Solutions:**
Can we do something computational to identify these? Maybe not worth spending time on, as this only likely occurs in surfaces exposed for a long time.


