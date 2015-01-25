#include "hmp.h"
#include <Eigen/Core>
#include <cstring>
#include <stdexcept>
#include <fstream>

void onlineclust::HMP::load2Dcts(const char* layer1, const char*layer2, const char* type)
{
  if(!strcmp(type, "rgb")){
    loadDct(layer1, 75, 150, this->D1rgb);
    loadDct(layer2, 2400, 1000, this->D2rgb);
  } else if(!strcmp(type, "depth")){
    loadDct(layer1, 25, 75, this->D1depth);
    loadDct(layer2, 1200, 500, this->D2depth);
  } else if(!strcmp(type, "normal")){
    loadDct(layer1, 25, 150, this->D1normal);
    loadDct(layer2, 2400, 1000, this->D2normal);
  } else if(!strcmp(type, "gray")){
    loadDct(layer1, 25, 75, this->D1normal);
    loadDct(layer2, 1200, 1000, this->D2normal);
  } else {
    std::cerr << "Unknown dictionary type!\n";
  }
}

void onlineclust::HMP::hmp_core(Eigen::MatrixXd& X, const char* type, uint SPlevel[2], Eigen::MatrixXd& fea)
{
  uint nchannel = (!strcmp(type,"rgb"))? 3 : 1;
  
  //matSize imSize =  matSize(X.rows(), X.cols()/3);
  uint num_w = ceil((float)(X.cols()/nchannel - (uint)(patchsz.width/2) * 2)/(float)stepsz.width1);
  uint num_h = ceil((float)(X.rows() - (uint)(patchsz.height/2) * 2)/(float)stepsz.height1);
  matSize gamma_sz =  matSize(num_w, num_h);

  Eigen::MatrixXd patchMat;
  mat2patch(X, type, gamma_sz, patchMat);

  // remove dc part of signal
  char str[] = "column";
  remove_dc(patchMat, str); 
  // 1st layer coding
  Eigen::MatrixXd Gamma; 
  
  if(nchannel == 3)
    omp::Batch_OMP(patchMat, D1rgb, SPlevel[0], Gamma);
  else 
    omp::Batch_OMP(patchMat, D1depth, SPlevel[0], Gamma);
  // convert to non negative value
  Gamma = Gamma.cwiseAbs();
  
  matSize psz1 = matSize(4,4);
  Eigen::MatrixXd omp_pool;
  uint fea_sz[2];
  MaxPool_layer1_mode1(Gamma, psz1, gamma_sz, omp_pool, fea_sz);

  // normalize with threshold
  double threshold = 0.1;
  for(uint i = 0; i < omp_pool.cols(); ++i){
    double norm = omp_pool.col(i).norm();
    if(norm < threshold) norm = 0.1;
    omp_pool.col(i) /= norm;
  }

    // 2nd layer learning
  Eigen::MatrixXd fea_temp;
  if(nchannel == 3)
    omp::Batch_OMP(omp_pool, D2rgb, SPlevel[1], fea_temp);
  else
    omp::Batch_OMP(omp_pool, D2depth, SPlevel[1], fea_temp);

  //if(nchannel==1)
  //  std::cout << fea_temp.norm() << std::endl;
  // absolut feature
  fea_temp = fea_temp.cwiseAbs();  
  // uint ct = 0;
  // for(uint j = 0; j < fea.rows();++j)
  //   for(uint i = 0; i < fea.cols();++i)
  //     if(fea(j,i)!= 0.0)++ct;
  // std::cout << ct << std::endl;
  
  //matSize feaSize = matSize{gamma_sz.first/psz1.first, 
  //                             gamma_sz.second/psz1.second};
  
  uint pool[3] = {3,2,1}; 
  //MaxPool_layer2(fea, feaSize, pool);
  MaxPool_layer2(fea_temp, fea_sz, pool, fea);
}

void onlineclust::HMP::hmp_core_mode1(Eigen::MatrixXd& X, const char* type, uint SPlevel[2], Eigen::MatrixXd& fea)
{
  uint nchannel = (!strcmp(type,"rgb"))? 3 : 1;
  
  //matSize imSize =  matSize(X.rows(), X.cols()/3);
  uint num_w = ceil((float)(X.cols()/nchannel - (uint)(patchsz.width/2) * 2)/(float)stepsz.width1);
  uint num_h = ceil((float)(X.rows() - (uint)(patchsz.height/2) * 2)/(float)stepsz.height1);
  matSize gamma_sz =  matSize(num_w, num_h);

  Eigen::MatrixXd patchMat;
  mat2patch(X, type, gamma_sz, patchMat);
  
  // remove dc part of signal
  char str[] = "column";
  remove_dc(patchMat, str); 

  // 1st layer coding
  Eigen::MatrixXd Gamma; 
  
  if(nchannel == 3)
    omp::Batch_OMP(patchMat, D1rgb, SPlevel[0], Gamma);
  else 
    omp::Batch_OMP(patchMat, D1depth, SPlevel[0], Gamma);
  // convert to non negative value
  Gamma = Gamma.cwiseAbs();
  
  matSize psz1 = matSize(4,4);
  Eigen::MatrixXd omp_pool;
  uint fea_sz[2];
  MaxPool_layer1_mode1(Gamma, psz1, gamma_sz, omp_pool, fea_sz);

  // normalize with threshold
  double threshold = 0.1;
  for(uint i = 0; i < omp_pool.cols(); ++i){
    double norm = omp_pool.col(i).norm();
    if(norm < threshold) norm = 0.1;
    omp_pool.col(i) /= norm;
  }

    // 2nd layer learning
  Eigen::MatrixXd fea_temp;
  if(nchannel == 3)
    omp::Batch_OMP(omp_pool, D2rgb, SPlevel[1], fea_temp);
  else
    omp::Batch_OMP(omp_pool, D2depth, SPlevel[1], fea_temp);

  // absolut feature
  fea_temp = fea_temp.cwiseAbs();  
  
  uint pool[3] = {3,2,1}; 
  // max_pooling in layer2
  uint sum = pow(pool[0],2) + pow(pool[1],2) + pow(pool[2],2);
  fea = Eigen::MatrixXd::Zero( sum * (fea_temp.rows() + Gamma.rows()), 1);
  Eigen::MatrixXd fea2, fea1;
  MaxPool_layer2(fea_temp, fea_sz, pool, fea2);

  // max pool features from first layer
  uint gammaSize[2] = {num_h, num_w};
  uint poolrate[2] = {4,4};
  max_pooling(Gamma, gammaSize, poolrate);

  for(uint i = 0; i < Gamma.cols(); ++i){
    double norm = Gamma.col(i).norm();
    if(norm < threshold) norm = 0.1;
    Gamma.col(i) /= norm;
  }
  MaxPool_layer2(Gamma, fea_sz, pool, fea1);
  
  fea.block(0,0, sum*fea_temp.rows(),1) = fea2;
  fea.block(sum*fea_temp.rows(),0, sum*Gamma.rows(),1) = fea1;

  fea /= fea.norm() + eps;
}

void onlineclust::HMP::hmp_core_mode2(Eigen::MatrixXd& X, const char* type, uint SPlevel[2], Eigen::MatrixXd& fea)
{
  uint nchannel = (!strcmp(type,"rgb"))? 3 : 1;
  
  //matSize imSize =  matSize(X.rows(), X.cols()/3);
  uint num_w = ceil((float)(X.cols()/nchannel - (uint)(patchsz.width/2) * 2)/(float)stepsz.width1);
  uint num_h = ceil((float)(X.rows() - (uint)(patchsz.height/2) * 2)/(float)stepsz.height1);
  matSize gamma_sz =  matSize(num_w, num_h);

  Eigen::MatrixXd patchMat;
  mat2patch(X, type, gamma_sz, patchMat);
  
  // remove dc part of signal
  char str[] = "column";
  remove_dc(patchMat, str); 

  // 1st layer coding
  Eigen::MatrixXd Gamma; 
  
  if(nchannel == 3)
    omp::Batch_OMP(patchMat, D1rgb, SPlevel[0], Gamma);
  else 
    omp::Batch_OMP(patchMat, D1depth, SPlevel[0], Gamma);
  // convert to non negative value
  Gamma = Gamma.cwiseAbs();
  
  // matSize psz1 = matSize(4,4);
  // Eigen::MatrixXd omp_pool;
  uint fea_sz[2];
  fea_sz[1] = ceil((num_w - 4 + 1)/4);
  fea_sz[0] = ceil((num_h - 4 + 1)/4);


  // MaxPool_layer1_mode1(Gamma, psz1, gamma_sz, omp_pool, fea_sz);

  // normalize with threshold
  // for(uint i = 0; i < omp_pool.cols(); ++i){
  //   double norm = omp_pool.col(i).norm();
  //   if(norm < threshold) norm = 0.1;
  //   omp_pool.col(i) /= norm;
  // }

  //   // 2nd layer learning
  // Eigen::MatrixXd fea_temp;
  // if(nchannel == 3)
  //   omp::Batch_OMP(omp_pool, D2rgb, SPlevel[1], fea_temp);
  // else
  //   omp::Batch_OMP(omp_pool, D2depth, SPlevel[1], fea_temp);

  // // absolut feature
  // fea_temp = fea_temp.cwiseAbs();  
  
  uint pool[3] = {3,2,1}; 
  // // max_pooling in layer2
  //uint sum = pow(pool[0],2) + pow(pool[1],2) + pow(pool[2],2);
  //fea = Eigen::MatrixXd::Zero( sum * Gamma.rows(), 1);
  // Eigen::MatrixXd fea2, fea1;
  // MaxPool_layer2(fea_temp, fea_sz, pool, fea2);

  // max pool features from first layer
  uint gammaSize[2] = {num_h, num_w};
  uint poolrate[2] = {4,4};
  max_pooling(Gamma, gammaSize, poolrate);

  double threshold = 0.1;
  for(uint i = 0; i < Gamma.cols(); ++i){
    double norm = Gamma.col(i).norm();
    if(norm < threshold) norm = 0.1;
    Gamma.col(i) /= norm;
  }
  MaxPool_layer2(Gamma, fea_sz, pool, fea);
  
  fea /= fea.norm() + eps;
}


void onlineclust::HMP::MaxPool_layer2(Eigen::MatrixXd &ifea, uint sz[2], uint pool[3], Eigen::MatrixXd &ofea)
{
  uint feaSize = ifea.rows();

  // reshape fea to mat form
  Eigen::MatrixXd temp_fea{sz[0], sz[1]*feaSize};
  // copy to temp_fea
  uint cols = 0;
  for(uint i = 0; i < sz[1]; ++i)
    for(uint j = 0; j < sz[0]; ++j){
      temp_fea.block(j, i*feaSize, 1, feaSize) = ifea.col(cols).transpose();
      ++cols;
    }

  // compute final feature size
  uint ofsize = pow(pool[0],2) + pow(pool[1],2) + pow(pool[2],2);
  ofea = Eigen::MatrixXd{ofsize * feaSize,1};

  // pool[0] = 3
  // uint pnw = ceil(((float)sz[1] - 0.5) / (float)pool[0]);
  // uint pnh = ceil(((float)sz[0] - 0.5) / (float)pool[0]);
  uint pnw = ceil((float)sz[1] / (float)pool[0]);
  uint pnh = ceil((float)sz[0] / (float)pool[0]);

  // pnw = std::max(pnw, sz[1] - (pool[0]-1) * pnw);
  // pnh = std::max(pnh, sz[0] - (pool[0]-1) * pnh);
  uint spw = 0, sph = 0;
  uint nfea = 0;

  for(uint i = 0; i < pool[0]; ++i)
    for(uint j = 0; j < pool[0]; ++j){    
      spw = i * pnw;
      sph = j * pnh;

      Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(feaSize, pnw*pnh);

      uint cols = 0;
      for(uint m = 0; m < pnw; ++m)
	for(uint l = 0; l < pnh; ++l){
	  uint h = sph+l;
	  uint w = spw+m;
	  if(w<sz[1] && h<sz[0]){
	    temp.col(cols) = temp_fea.block(h, w*feaSize,1, feaSize).transpose();
	    ++cols;
	  }
	}
      ofea.block(feaSize*nfea, 0, feaSize, 1)
	= temp.rowwise().maxCoeff();
      ++nfea;
    }

  // pool[1] = 2
  pnw = ceil((float)sz[1] / (float)pool[1]);
  pnh = ceil((float)sz[0] / (float)pool[1]);

  spw = 0; sph = 0;

  for(uint i = 0; i < pool[1]; ++i)
    for(uint j = 0; j < pool[1]; ++j){    
      spw = i * pnw;
      sph = j * pnh;

      Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(feaSize, pnw*pnh);

      uint cols = 0;
      for(uint m = 0; m < pnw; ++m)
	for(uint l = 0; l < pnh; ++l){
	  uint h = sph+l;
	  uint w = spw+m;
	  if(w<sz[1] && h<sz[0]){
	    temp.col(cols) = temp_fea.block(h, w*feaSize,1, feaSize).transpose();
	    ++cols;
	  }
	}
      ofea.block(feaSize*nfea, 0, feaSize, 1)
	= temp.rowwise().maxCoeff();
      ++nfea;
    }

  // pool[2] = 1
  //ofea.block(feaSize*nfea, 0, feaSize, 1) = temp1.rowwise().maxCoeff();
  ofea.block(feaSize*nfea, 0, feaSize, 1) = ifea.rowwise().maxCoeff();
  //std::cout << ifea.norm() << std::endl;
  //normalize
  ofea /= ofea.norm() + eps;
  
}

// void onlineclust::HMP::MaxPool_layer2(Eigen::MatrixXd &fea, matSize const&feaSize, uint pool[3])
// { 
//   uint rows = fea.rows();
//   //uint cols = fea.cols();
//   Eigen::MatrixXd temp =std::move(fea);
//   uint nr = pow(pool[0],2) + pow(pool[1],2) + pow(pool[2],2);
//   fea = Eigen::MatrixXd::Zero(static_cast<int>(nr*rows),1);
  
//   // pool 1
//   Eigen::MatrixXd maxfea{rows, 16};
//   uint psw = feaSize.first / pool[0];
//   uint psh = feaSize.second / pool[0];
//   uint blocksz = psw * psh;
//   Eigen::MatrixXd max_tmp{rows, blocksz};
  
//   for(uint j = 0; j < pool[0]; ++j )
//     for(uint i = 0; i < pool[0]; ++i){
//       uint spw = i * psw;
//       uint sph = j * psh;
      
//       for(uint l = 0; l < psh; ++l){
// 	max_tmp.block(0, l*psw, rows,psw) = temp.block(0,spw + (sph+l)*feaSize.first, rows, psw);
//       }
//       fea.block((i+j*pool[0])*1000, 0, 1000,1) = max_tmp.rowwise().maxCoeff();
//     }

//   // pool 2
//   psw = feaSize.first / pool[1];
//   psh = feaSize.second / pool[1];
//   blocksz = psw * psh;
//   max_tmp = Eigen::MatrixXd{rows, blocksz};
//   uint offset = pool[0] * pool[0];

//   for(uint j = 0; j < pool[1]; ++j )
//     for(uint i = 0; i < pool[1]; ++i){
//       uint spw = i * psw;
//       uint sph = j * psh;
      
//       for(uint l = 0; l < psh; ++l)
// 	max_tmp.block(0, l*psw, rows,psw) = temp.block(0,spw + (sph+l)*feaSize.first, rows, psw);

//       fea.block((offset+i+j*pool[1])*1000, 0, 1000,1) = max_tmp.rowwise().maxCoeff();
//     }

//   // pool 3
//   uint offset2 = offset + pool[1]*pool[1];
//   fea.block(offset2*1000, 0, 1000, 1) = fea.block(offset*1000, 0, 1000,1);
//   for(uint j = 0; j < pool[1]; ++j )
//     for(uint i = 1; i < pool[1]; ++i){
//       fea.block(offset2*1000, 0, 1000, 1) = fea.block(offset2*1000, 0, 1000, 1).cwiseMax(fea.block((offset+i+j*pool[1])*1000, 0, 1000,1));
//     }
  
//   // normalize feature
//   fea /= (fea.norm() + eps);

// }

void onlineclust::HMP::MaxPool_layer1_mode1(Eigen::MatrixXd const&Gamma, matSize const&psz, matSize const &realsz, Eigen::MatrixXd &omp_pooling, uint sz[2])
{
  uint feaSize = Gamma.rows();
  
  uint nw = floor(realsz.first / psz.first);
  uint nh = floor(realsz.second / psz.second);

  Eigen::MatrixXd omp_pooling_temp = Eigen::MatrixXd::Zero(nh, nw * feaSize); 
  
  uint spw = 0, sph = 0;
  Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(feaSize, psz.second*psz.first);
  uint colums = 0;

  for(uint i = 0; i < nw; ++i)
    for(uint j = 0; j < nh; ++j){
    
      spw = i * psz.first;
      sph = j * psz.second;

      uint cols = 0;
      for(uint m = 0; m < psz.first; ++m)
	for(uint l = 0; l < psz.second; ++l){	
	  temp.col(cols) = Gamma.col( (sph+l) + (spw+m)*realsz.second );
	  ++cols;
	}
      omp_pooling_temp.block(j, i*feaSize, 1, feaSize)
	= temp.rowwise().maxCoeff().transpose();
      ++colums;
    }
  
  uint dim[] = {nh, nw, feaSize};
  uint patchsz[] = {psz.second, psz.first};
  uint stepsz[] = {1,1,1};
  
  mat2col(omp_pooling_temp, dim, patchsz, stepsz, omp_pooling, sz);
  
}

void onlineclust::HMP::mat2patch(Eigen::MatrixXd const& im, const char*type, matSize const& rsz, Eigen::MatrixXd &patchMat)
{
  uint nchannel;
  if(!strcmp(type, "rgb")){
    nchannel = 3;
  } else if (!strcmp(type, "depth")){
    nchannel = 1;
  } else {
    throw std::runtime_error("\nUnknow type!!!\n");
  }

  uint nPatchx = rsz.first; 
  uint nPatchy = rsz.second;

  patchMat = Eigen::MatrixXd{patchsz.width * patchsz.height * nchannel,
		   nPatchx * nPatchy};
  
  uint cols = 0, srow, scol;
  int npx = patchsz.height * patchsz.width;

 
  for(uint j = 0; j < nPatchy; ++j){
     for(uint i = 0; i < nPatchx; ++i){
       
      scol = i * stepsz.width1 * nchannel;
      srow = j * stepsz.height1;

      // copy to output matrix patch2dMat, order r,g,b
      for(uint m = 0; m < patchsz.height; ++m)
	for(uint l = 0; l < patchsz.width; ++l)
	  for(uint ch = 0; ch < nchannel; ++ch)
	    {
	      uint pos = nchannel - 1 - ch;
	      patchMat(m + l*patchsz.height + ch*npx, cols)
		= im(srow + m, scol + l*nchannel + pos);
	    }
      ++cols;
    }
  }      

}

void onlineclust::HMP::mat2col(Eigen::MatrixXd const& mat, uint dim[3], uint patchsz[2], uint stepsz[3], Eigen::MatrixXd &matcol, uint sz[2])
{
  uint patch_nw = ceil((dim[1] - patchsz[1] + 1)/stepsz[1]);
  uint patch_nh = ceil((dim[0] - patchsz[0] + 1)/stepsz[0]);

  sz[0] = patch_nh; sz[1] = patch_nw;
  
  matcol = Eigen::MatrixXd{dim[2]*patchsz[0]*patchsz[1], patch_nw*patch_nh};
  
  uint cols = 0;
  for(uint i = 0; i < patch_nw; ++i)
    for(uint j = 0; j < patch_nh; ++j){
      uint sph = j * stepsz[0];
      uint spw = i * stepsz[1];

      for(uint k = 0; k < dim[2]; ++k)
	for(uint l = 0; l < patchsz[1]; ++l)
	  for(uint m = 0; m < patchsz[0]; ++m){

	    matcol( k*patchsz[0]*patchsz[1] + l*patchsz[0] + m, cols)
	      = mat(sph+m, (spw+l)*dim[2] + k);
	  
	  }
      ++cols;
    }

}

void onlineclust::HMP::max_pooling(Eigen::MatrixXd& mat, uint sz[2], uint pool[2])
{
  Eigen::MatrixXd temp = std::move(mat);
  
  uint feaSize = temp.rows();
  
  uint nw = floor(sz[1] / pool[1]);
  uint nh = floor(sz[0] / pool[1]);

  mat = Eigen::MatrixXd::Zero(feaSize,nh*nw); 
  
  uint spw = 0, sph = 0;
  Eigen::MatrixXd tempMax = Eigen::MatrixXd::Zero(feaSize, pool[0]*pool[1]);
  uint colums = 0;

  for(uint i = 0; i < nw; ++i)
    for(uint j = 0; j < nh; ++j){
    
      spw = i * pool[1];
      sph = j * pool[0];

      uint cols = 0;
      for(uint m = 0; m < pool[1]; ++m)
	for(uint l = 0; l < pool[0]; ++l){	
	  tempMax.col(cols) = temp.col( (sph+l) + (spw+m)*sz[0] );
	  ++cols;
	}
      mat.col(colums) = tempMax.rowwise().maxCoeff();
      ++colums;
    }
  
}
