#include<iostream>
#include<Eigen/Dense>
#include<string>
#include<vector>
#include<stdlib.h>
#include "system.h"
#include "analysis.h"
#include "coshfunc.h"
#include "emulator.h"
#include "mcmc.h"

int main(int argc, char* argv[])
{
  if(argc!=3)
    {
      printf("Usage: ./parameters start finish\n");
      return 1;
    }

  std::string 
    foldername="model_output",
    writefilename="parameters.dat",
    rangename=foldername+"/parameter_priors.dat",
    infilename=foldername+"/parameters_lch.dat",
    gabname=foldername+"/gabfunctions.dat",
    delimiter=" ";
  //start, finish for model runs to use
  int
    start=atoi(argv[1]),
    finish=atoi(argv[2]),
    ab=4;
  Eigen::MatrixXd Parameters;

  WriteParameterFiles(rangename, foldername, writefilename, delimiter,
		      start, finish, ab, 
		      Parameters);
  WriteFile(infilename,Parameters,delimiter);
  WriteGABFunctions(gabname,Parameters,delimiter,ab);
  std::string cmd = "for((i="+std::to_string(start)+";i<"+std::to_string(finish)+";i++)); do fn=$(printf '"+foldername+"/run%04d/' $i); cp -v "+foldername+"/fixed_parameters.dat $fn; done";
  system(cmd.c_str());
}
