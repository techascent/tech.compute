package tech.compute.cpu;


public interface TypedUnaryOp
{
  public final int UNSIGNED = 1;
  public final int FLOAT = 1 << 1;

  double doubleOp(double lhs, int flags);
  float floatOp(float lhs, int flags);
  long longOp(long lhs, int flags);
  int intOp(int lhs, int flags);
  short shortOp(short lhs, int flags);
  byte byteOp(byte lhs, int flags);
}
