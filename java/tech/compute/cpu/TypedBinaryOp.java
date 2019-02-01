package tech.compute.cpu;


public interface TypedBinaryOp
{
  public final int UNSIGNED = 1;
  public final int FLOAT = 1 << 1;

  
  double doubleOp(double lhs, double rhs, int flags);
  float floatOp(float lhs, float rhs, int flags);
  long longOp(long lhs, long rhs, int flags);
  int intOp(int lhs, int rhs, int flags);
  short shortOp(short lhs, short rhs, int flags);
  byte byteOp(byte lhs, byte rhs, int flags);
};
