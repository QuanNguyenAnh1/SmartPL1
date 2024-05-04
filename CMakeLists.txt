cmake_minimum_required(VERSION 3.0)
##############################
PROJECT( car-parking )
##############################

set(OpenCV_DIR "C:/opencv/build")
# Packages
find_package( OpenCV REQUIRED )


# Includes
include_directories(	
	${OPENCV_INCLUDE_DIR}
	${PROJECT_BINARY_DIR}
)

set(SOURCE_FILES 
	main.cpp
	IPM.cpp
	IPM.h
	ImageProcessor.h
	ImageProcessor.cpp
)


# Add executable and target link libraries
ADD_EXECUTABLE( car-parking ${SOURCE_FILES})
TARGET_LINK_LIBRARIES( car-parking ${OpenCV_LIBS})


