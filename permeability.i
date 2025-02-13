%module permeability
%include "std_vector.i"
%{
double getPermeability(const std::vector<double>&, int, int, int);
%}
%template(dvector) std::vector<double>;
%include "swig/permeability.h"
