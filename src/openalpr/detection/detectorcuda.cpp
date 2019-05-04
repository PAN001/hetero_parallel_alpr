/*
 * Copyright (c) 2013 OpenALPR Technology, Inc.
 * Open source Automated License Plate Recognition [http://www.openalpr.com]
 *
 * This file is part of OpenALPR.
 *
 * OpenALPR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License
 * version 3 as published by the Free Software Foundation
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "detectorcuda.h"
#include <omp.h>
#ifdef COMPILE_GPU

using namespace cv;
using namespace std;


namespace alpr
{

  DetectorCUDA::DetectorCUDA(Config* config, PreWarp* prewarp) : Detector(config, prewarp) {


#if OPENCV_MAJOR_VERSION == 2
	      std::cout << "RRRRRRRRRRRRRRRRRRRRRRR " << OPENCV_MAJOR_VERSION << std::endl;

//    if( this->cuda_cascade.load( get_detector_file() ) )
      if (true)
#else
    cuda_cascade = cuda::CascadeClassifier::create(get_detector_file());
    if( !this->cuda_cascade.get()->empty() )
#endif
    {
      this->loaded = true;
      printf("--(!)Loaded CUDA classifier\n");
   
      omp_set_num_threads(4);  
      #pragma omp parallel num_threads(2)
      {
          int tid = omp_get_thread_num();
          printf("Hello World from thread = %d\n", tid);

          cv::gpu::CascadeClassifier_GPU cuda_cascade12;
	  if (tid == 0) {
	      this->cuda_cascade0.load( get_detector_file() ) ;
	  } else if (tid == 1) {
	      this->cuda_cascade1.load( get_detector_file() ) ;
	  }
      }
      std::cout << "threads : " << omp_get_num_threads() << endl;
    }
    else
    {
      this->loaded = false;
      printf("--(!)Error loading CPU classifier %s\n", get_detector_file().c_str());
    }
  }


  DetectorCUDA::~DetectorCUDA() {
  }

  vector<Rect> DetectorCUDA::find_plates(Mat frame, cv::Size min_plate_size, cv::Size max_plate_size)
  {
    //-- Detect plates
    vector<Rect> plates;
    
    timespec startTime;
    getTimeMonotonic(&startTime);
#if OPENCV_MAJOR_VERSION == 2
#else
    cuda::GpuMat cudaFrame, plateregions_buffer;
#endif
#if OPENCV_MAJOR_VERSION == 2
    /*int numdetected = cuda_cascade.detectMultiScale(cudaFrame, plateregions_buffer, 
            (double) config->detection_iteration_increase, config->detectionStrictness, 
            min_plate_size); */
    //int numdetected0, numdetected1;  
    #pragma omp parallel num_threads(2)
    {
          int tid = omp_get_thread_num();
	  int numdetected = 0;
	  gpu::GpuMat plateregions_buffer;
    	  if (tid == 0) {	 
            numdetected = cuda_cascade0.detectMultiScale(frame, plateregions_buffer,
            (double) config->detection_iteration_increase, config->detectionStrictness,
            min_plate_size);
	    std::cout << "------ thread: " << tid << " numberdetect: " << numdetected << endl;
	  } else {
	    numdetected = cuda_cascade1.detectMultiScale(frame, plateregions_buffer,
	    (double) config->detection_iteration_increase, config->detectionStrictness,
            min_plate_size);
	    std::cout << "------ thread: " << tid << " numberdetect: " << numdetected << endl;
	  }

	if (numdetected > 0) {
	    Mat plateregions_downloaded;
	    plateregions_buffer.colRange(0, numdetected).download(plateregions_downloaded);
	    #pragma omp critical 
	    {
	        for (int i=0; i<numdetected; i++)
		    plates.push_back(plateregions_downloaded.ptr<cv::Rect>()[i]);
	    }
     	}
    }
	
#else
    cuda_cascade->setScaleFactor((double) config->detection_iteration_increase);
    cuda_cascade->setMinNeighbors(config->detectionStrictness);
    cuda_cascade->setMinObjectSize(min_plate_size);
	cuda_cascade->detectMultiScale(cudaFrame,
			plateregions_buffer);
	std::vector<Rect> detected;
	cuda_cascade->convert(plateregions_buffer, detected);
	int numdetected = detected.size();
#endif
  
  /*  std::cout << "------ number detected= " << numdetected0 << " " << numdetected1 << std::endl; 
 */
    //plateregions_buffer0.colRange(0, numdetected0).download(plateregions_downloaded0);
    //plateregions_buffer1.colRange(0, numdetected1).download(plateregions_downloaded1);
/*
    for (int i = 0; i < numdetected0; ++i)
    {
      plates.push_back(plateregions_downloaded0.ptr<cv::Rect>()[i]);
    }
	
    for (int i = 0; i < numdetected1; ++i)
    {
      plates.push_back(plateregions_downloaded1.ptr<cv::Rect>()[i]);
    }
*/
    if (config->debugTiming)
    {
      timespec endTime;
      getTimeMonotonic(&endTime);
      cout << "LBP Time: " << diffclock(startTime, endTime) << "ms." << endl;
    }
    
    return plates;
  }

}

#endif
