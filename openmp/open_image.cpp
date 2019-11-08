// FILE			: open_image.cpp
// AUTHOR		: K.Supasan
// CREATE ON	: 2019, November 07 (UTC+0)
// MAINTAINER	: K.Supasan

// MACRO DETAIL

// README

// REFERENCE

// MACRO SET

// MACRO CONDITION

#include    <opencv2/core.hpp>
#include    <opencv2/highgui.hpp>
#include    <opencv2/imgcodecs.hpp>
#include    <iostream>

using namespace cv;
using namespace std;
int main( int argc, char** argv )
{
    String imageName( "../data/HappyFish.jpg" ); // by default
    if( argc > 1)
    {
        imageName = argv[1];
    }
    Mat image;
    image = imread( imageName, IMREAD_COLOR ); // Read the file
    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image );                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
