# SSA_SST_DETECTION
Alormar Camera Analysis code for satellite tracking.

There are some important things to note about the files of this Project.

Main Code:
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
    the test of this algorithom and was used to try different methods of how you can extract 
    the satellites speed and height from a 2d image.
    
  - Filter Tests Code is partly implemented in the Detection Code. The Filter test was a try what would
    be possible to detect in an image. In its current setup can find satellites that are not visible to
    the human eye, but the filter is so sensitive it will be very prone to error due to changing light conditions
    and it will probably not work on different cameras (not tested)
   
  Objects:
    
  - There are a view objects, they mainly store data but handle some functions as well.
  
  - Satellite_oop Stores all satellites from one iteration. The main functionality is the positioning of the satellite
    using close stars from the Star Class.
    
  - Star_Img_oop (star Class) stores all image detcted stars ans translates thier positions to horizontal coordinates. If a star is 
    successfully intentified this info will be stored inside the class, as well as its magnitude and size in the image
    
  - Star_calc_2_oop is the class that handles all calculated stars e.g. stars from the database. If a star from the database
    is close to the satellite, its hozizontal coordinates will be calculated from the database. I would
    probably be possible to merge the star_calc_2 class and Star_Img class. 
    
  Database:
  - There are two types of databases i reference. A star and a satellite database. The star database has approx 120.000 stars
    that can be used to compare to stars seen in the image.
    
  - The satellite Database was used to make the development easier and to comapre and save results
    from the calculations. This database is a text file and stores the satellites seen in the image as
    a line. If a satellite is similar to one in the database it will be appended and identified as the same.
    
    
 Report:
  - The report is the formal summary of the project.
  
  - Development report: This is a file, were I documented the process of developing the algorithims and the code in more
    detail.

   
  
    
