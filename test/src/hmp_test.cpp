#include "io.h"
#include "proc.h"
#include "hmp.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <fstream>
#include <chrono>

using namespace onlineclust;
using namespace onlineclust::omp;
using namespace cv;
using namespace Eigen;
using namespace std;
using namespace engine;
using namespace std::chrono;

int main(){

  /** rgbd */
  HMP hmp("hmp.config", "rgbd","second+first");
  char rgbdir[] = "apple_1_1_1_crop.png";
  char depthdir[] = "apple_1_1_1_depthcrop.png";

  VectorXd fea;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  hmp.computeHMP(rgbdir, depthdir, fea);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  cout << "Time spent for HMP feature: "
       << duration_cast<milliseconds>(t2-t1).count()<<endl;

  /** rgb */
  // HMP hmp("hmp.config", "rgb","second+first");
  // char rgbdir[] = "apple_1_1_1_crop.png";

  // VectorXd fea;
  // high_resolution_clock::time_point t1 = high_resolution_clock::now();
  // hmp.computeHMP(rgbdir, fea);
  // high_resolution_clock::time_point t2 = high_resolution_clock::now();

  // cout << "Time spent for HMP feature: "
  // 	       << duration_cast<milliseconds>(t2-t1).count()<<endl;

  /** depth */
      
  // HMP hmp("hmp.config", "depth","second+first");
  // char depthdir[] = "apple_1_1_1_depthcrop.png";

  // VectorXd fea;
  // high_resolution_clock::time_point t1 = high_resolution_clock::now();
  // hmp.computeHMP(depthdir, fea);
  // high_resolution_clock::time_point t2 = high_resolution_clock::now();

  // cout << "Time spent for HMP feature: "
  // 	   << duration_cast<milliseconds>(t2-t1).count()<<endl;
      
  return 0;
}
