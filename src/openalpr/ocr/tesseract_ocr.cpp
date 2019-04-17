/*
 * Copyright (c) 2015 OpenALPR Technology, Inc.
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

#include "tesseract_ocr.h"
#include "config.h"

#include "segmentation/charactersegmenter.h"

#include <omp.h>

using namespace std;
using namespace cv;
using namespace tesseract;

namespace alpr
{

  TesseractOcr::TesseractOcr(Config* config)
  : OCR(config)
  {
    int i;
    const string MINIMUM_TESSERACT_VERSION = "3.03";
    this->postProcessor.setConfidenceThreshold(config->postProcessMinConfidence, config->postProcessConfidenceSkipLevel);
    for(i = 0;i < 2;i++) {
        if (cmpVersion(tesseracts[i].Version(), MINIMUM_TESSERACT_VERSION.c_str()) < 0)
        {
          std::cerr << "Warning: You are running an unsupported version of Tesseract." << endl;
          std::cerr << "Expecting at least " << MINIMUM_TESSERACT_VERSION << ", your version is: " << tesseracts[i].Version() << endl;
        }

        string TessdataPrefix = config->getTessdataPrefix();
        if (cmpVersion(tesseracts[i].Version(), "4.0.0") >= 0)
          TessdataPrefix += "tessdata/";    

        std::cout << "TessdataPrefix: " << TessdataPrefix << std::endl;
        std::cout << "config->ocrLanguage.c_str(): " << config->ocrLanguage.c_str() << std::endl;
        // Tesseract requires the prefix directory to be set as an env variable
        tesseracts[i].Init(TessdataPrefix.c_str(), config->ocrLanguage.c_str()  );
        tesseracts[i].SetVariable("save_blob_choices", "T");
        tesseracts[i].SetVariable("debug_file", "/dev/null");
        tesseracts[i].SetPageSegMode(PSM_SINGLE_CHAR);
    }

    // if (cmpVersion(tesseract.Version(), MINIMUM_TESSERACT_VERSION.c_str()) < 0)
    // {
    //   std::cerr << "Warning: You are running an unsupported version of Tesseract." << endl;
    //   std::cerr << "Expecting at least " << MINIMUM_TESSERACT_VERSION << ", your version is: " << tesseract.Version() << endl;
    // }

    // string TessdataPrefix = config->getTessdataPrefix();
    // if (cmpVersion(tesseract.Version(), "4.0.0") >= 0)
    //   TessdataPrefix += "tessdata/";    

    // // Tesseract requires the prefix directory to be set as an env variable
    // tesseract.Init(TessdataPrefix.c_str(), config->ocrLanguage.c_str()  );
    // tesseract.SetVariable("save_blob_choices", "T");
    // tesseract.SetVariable("debug_file", "/dev/null");
    // tesseract.SetPageSegMode(PSM_SINGLE_CHAR);
  }

  TesseractOcr::~TesseractOcr()
  {
    // int i;
    // for(i = 0;i < 1;i++) {
    //   tesseracts[i].End();
    // }
  }
  
  std::vector<OcrChar> TesseractOcr::recognize_line(int line_idx, PipelineData* pipeline_data) {
    std::cout << "========================== TesseractOcr::recognize_line: line_idx = " << line_idx << " ==========================" << std::endl;
    const int SPACE_CHAR_CODE = 32;
    
    std::vector<OcrChar> recognized_chars;
    
    std::cout << "========================== pipeline_data->thresholds.size(): " << pipeline_data->thresholds.size() << " ==========================" << std::endl;
    std::cout << "========================== pipeline_data->charRegions[line_idx].size(): " << pipeline_data->charRegions[line_idx].size() << " ==========================" << std::endl;
    // TODO：可parallel char加入顺序貌似无所谓
    int thread_count = 2;
    omp_set_nested(1);
    omp_set_dynamic(0);
    omp_set_num_threads(thread_count);
    // #pragma omp parallel for num_threads(thread_count)
    #pragma omp parallel for schedule(static)
    // #pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < pipeline_data->thresholds.size(); i++)
    {
      int thread_id = omp_get_thread_num();
      // int thread_id = 0;
      printf("thread_id: %d i:%d \n", thread_id, i);


      tesseract::TessBaseAPI& tesseract = tesseracts[thread_id];
      std::cout << thread_id << " " << "DEBUG: 0" << std::endl;

      // Make it black text on white background
      bitwise_not(pipeline_data->thresholds[i], pipeline_data->thresholds[i]);
      std::cout << thread_id << " " << "DEBUG: 0-1" << std::endl;
      tesseract.SetImage((uchar*) pipeline_data->thresholds[i].data, 
                          pipeline_data->thresholds[i].size().width, pipeline_data->thresholds[i].size().height, 
                          pipeline_data->thresholds[i].channels(), pipeline_data->thresholds[i].step1());

      std::cout << thread_id << " " << "DEBUG: 1" << std::endl;
      // int absolute_charpos = 0;
      #pragma omp parallel for schedule(static)
      for (unsigned int j = 0; j < pipeline_data->charRegions[line_idx].size(); j++)
      {
        int absolute_charpos = j;
        // Trace: iteratre through each segemented char (region): there might be several chars in one region but idealy only one
        Rect expandedRegion = expandRect( pipeline_data->charRegions[line_idx][j], 2, 2, pipeline_data->thresholds[i].cols, pipeline_data->thresholds[i].rows) ;

        tesseract.SetRectangle(expandedRegion.x, expandedRegion.y, expandedRegion.width, expandedRegion.height);
        std::cout << thread_id << " " << "DEBUG: 1-1" << std::endl;
        tesseract.Recognize(NULL); // TODO: recognize

        std::cout << thread_id << " " << "DEBUG: 2" << std::endl;

        tesseract::ResultIterator* ri = tesseract.GetIterator();
        std::cout << thread_id << " " << "DEBUG: 3" << std::endl;
        tesseract::PageIteratorLevel level = tesseract::RIL_SYMBOL;
        int ri_cnt = 0;
        // TODO: parallel around do...while
        do
        {
          const char* symbol = ri->GetUTF8Text(level);
          float conf = ri->Confidence(level);

          bool dontcare;
          int fontindex = 0;
          int pointsize = 0;
          const char* fontName = ri->WordFontAttributes(&dontcare, &dontcare, &dontcare, &dontcare, &dontcare, &dontcare, &pointsize, &fontindex);

          std::cout << thread_id << " " << "DEBUG: 4" << std::endl;
          // Ignore NULL pointers, spaces, and characters that are way too small to be valid
          if(symbol != 0 && symbol[0] != SPACE_CHAR_CODE && pointsize >= config->ocrMinFontSize)
          {
            OcrChar c;
            c.char_index = absolute_charpos;
            c.confidence = conf;
            c.letter = string(symbol);
            recognized_chars.push_back(c);

            if (this->config->debugOcr)
              printf("charpos%d line%d: ri_cnt %d: threshold %d:  symbol %s, conf: %f font: %s (index %d) size %dpx", absolute_charpos, line_idx, ri_cnt, i, symbol, conf, fontName, fontindex, pointsize);

            bool indent = false;
            tesseract::ChoiceIterator ci(*ri);
            do
            {
              const char* choice = ci.GetUTF8Text();
              
              OcrChar c2;
              c2.char_index = absolute_charpos;
              c2.confidence = ci.Confidence();
              c2.letter = string(choice);
              
              //1/17/2016 adt adding check to avoid double adding same character if ci is same as symbol. Otherwise first choice from ResultsIterator will get added twice when choiceIterator run.
              if (string(symbol) != string(choice))
                recognized_chars.push_back(c2);
              else
              {
                // Explictly double-adding the first character.  This leads to higher accuracy right now, likely because other sections of code
                // have expected it and compensated. 
                // TODO: Figure out how to remove this double-counting of the first letter without impacting accuracy
                recognized_chars.push_back(c2);
              }
              if (this->config->debugOcr)
              {
                if (indent) printf("\t\t ");
                printf("\t- ");
                printf("%s conf: %f\n", choice, ci.Confidence());
              }

              indent = true;
            }
            while(ci.Next());

          }

          if (this->config->debugOcr)
            printf("---------------------------------------------\n");

          delete[] symbol;

          ri_cnt++;
        }
        while((ri->Next(level)));

        delete ri;

        // absolute_charpos++;
      }
    }
    
    return recognized_chars;
  }
  void TesseractOcr::segment(PipelineData* pipeline_data) {

    CharacterSegmenter segmenter(pipeline_data);
    segmenter.segment();
  }


}
