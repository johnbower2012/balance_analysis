#include<iostream>
#include<Eigen/Dense>
#include<string>
#include<vector>
#include<stdlib.h>
#include "parametermap.h"
#include "system.h"
#include "coshfunc.h"

int main(int argc, char* argv[])
{
  if(argc!=3)
    {
      printf("Usage: ./parameters start finish\n");
      return 1;
    }

  CParameterMap MAP;
  CSystem SYSTEM;
  CDistCosh Dist;

  MAP.ReadParsFromFile("write_parameters.dat");

  std::string 
    model_foldername=MAP.getS("MODEL_FOLDER","model_output"),
    param_filename=MAP.getS("PARAM_FILE","parameters.dat"),
    fixed_filename=MAP.getS("PARAM_FIXED_FILE","fixed_parameters.dat"),
    prior_filename=MAP.getS("PARAM_PRIOR_FILE","parameter_priors.dat"),
    lhc_filename=MAP.getS("PARAMS_FILE","lhc_parameters.dat"),
    delimiter=" ";
  //start, finish for model runs to use
  int
    start=atoi(argv[1]),
    finish=atoi(argv[2]),
    ab=MAP.getI("QUARK_PAIRS",4),
    GRID=MAP.getI("GRID",100);
  double
    ETA_MAX=MAP.getD("ETA_MAX",8);
  bool
    CUT_G0=MAP.getB("CUT_G0",true),
    FIX_G0=MAP.getB("FIX_G0",true),
    CUT_WIDTH=MAP.getB("CUT_WIDTH",false),
    COMMON_WIDTH=MAP.getB("COMMON_WIDTH",false);

  Eigen::MatrixXd Parameters;
  Eigen::MatrixXd Priors;
  std::vector<std::string> Distributions, Names;

  SYSTEM.LoadParamFile(prior_filename, Distributions, Names, Priors);
  LHCSampling(Parameters,finish-start,Priors);
  if(FIX_G0){
    Parameters = Dist.FixG0(ab,Parameters);
  }
  if(COMMON_WIDTH){
    int ParamsPerPair=Parameters.cols()/ab;
    for(int pair=1;pair<ab;pair++){
      Parameters.col(pair*ParamsPerPair) = Parameters.col(0);
    }
  }
  SYSTEM.WriteParamFileLoop(param_filename,model_foldername,start,Names,Parameters);
  SYSTEM.WriteFile(lhc_filename,Parameters);

  std::string cmd = "for((i="+std::to_string(start)+";i<"+std::to_string(finish)+";i++)); do fn=$(printf '"+model_foldername+"/run%04d/' $i); cp -v "+model_foldername+"/"+fixed_filename+" $fn; done";
  system(cmd.c_str());
}
