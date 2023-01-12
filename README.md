# SSA_SST_DETECTION
Alormar Camera Analysis code for satellite tracking.

There are some important things to note about the files of this Project.

Code:
There are two main scripts, one for the calibration and one for the actual satellite detection:

  - The calibration code was intendet to be convertet to c code and run as an windows executable
    but there are still some flaws in the code and some improvements to be made.
    Its purpose was so I would be able to quickly recalibrate the camera and to make it easier 
    if you wanted to try different cameras with different lenses.
    
  - Detection Code is the main if you want to call it that way. It combines most of the things 
    I have done at my time on alomar, It is a prototype of what a running satellite detction on 
    a camera could look like. It finds satrs and satellites in an image (either from a 
    camera directly passing it to the code or from an existing dataset, like I did) and can automatically
    position the satellite using the backgreound stars. From this positioning it calculates the
    height, speed and orbital period of the satellite and stores it in a database (text file).
    Wich can be used to check if the satellite has already been idenitfyied.
    
    Quick note about this database, it stores the satellite as speed, velocity, heading direction and 
    orbital peroid, not in Orbital parameters, wich would be more desireable.
    
    This code should work on the allsky camera, but for a gimbaled camera some parts of this code 
    would need to be adjusted or exchanged completely. You would have to implement a 
    calibration of the image based on the motor positions.
    
  - Perspective Correction Code is 1:1 implemented in the Detection code, The correction code is 
    the test of this algorithom and was used to try different methods of hoy you can extract 
    the satellites speed and height from a 2d image.
    
  - Filter Tests Code is partly implemented in the Detection Code. The Filter test was a try what would
    be possible to detect in an image. In its current setup can find satellites that are not visible to
    the human eye. 
    
