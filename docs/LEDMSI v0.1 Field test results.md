# LEDMSI v0.1 Field test results
## Summary
This document summarises what we learned from the first field trials of the LEDMSI system in June 2022 at Ginnerup and Sorte Muld.
## Things to fix for this version
### The system is too slow
**The problem:** It takes nearly 5 minutes to acquire the full image stack. This is too long.  
**Solutions:**  
1. The code surrently intitialises the arduino that controls the LEDS for every wavelength. It does this via `PyFirmata`, which takes a few seconds to intialise the board. Fix this so that the arduino is intialised in the `__init__` method of the `CameraControl`class.  
2.  It takes a few seconds to write each image to file. Part of this may be due to [a known issue with the raspberry pi](https://forums.raspberrypi.com/viewtopic.php?t=245931) and external storage. This issue has a known fix. Further to applying this fix we'll test writing to RAM via `initramfs` and then copying this to disk after 
3. Currently we're using `libcamera`to handle the camera. This is
## Things to fix for the next version

