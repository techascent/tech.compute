package tech.compute.cpu;


public interface UnaryReduce
{
  double initialize(double first_value);
  double update(double accum, double next_value);
  double finalize(double accum, int num_elems);
};
