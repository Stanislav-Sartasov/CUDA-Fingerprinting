using System.Diagnostics;

namespace CUDAFingerprinting.Common
{
    [DebuggerDisplay("X={X}, Y={Y}")]
    public struct Minutia
    {
        public int X;
        public int Y;
        public double Angle;

        public static bool operator ==(Minutia m1, Minutia m2)
        {
            return m1.Angle == m2.Angle && m1.X == m2.X && m1.Y == m2.Y;
        }

        public static bool operator !=(Minutia m1, Minutia m2)
        {
            return !(m1 == m2);
        }
    }
}
