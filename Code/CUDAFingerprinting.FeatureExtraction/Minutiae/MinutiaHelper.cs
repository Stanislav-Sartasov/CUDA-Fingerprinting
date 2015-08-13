using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.FeatureExtraction.Minutiae
{
    public class MinutiaHelper
    {
        public static float NormalizeAngle(float angle)
        {
            float res = angle - (float)(Math.Floor(angle/(2*Math.PI))*2*Math.PI);
            
            return res;
        }

        public static Minutia NormalizeAngle(Minutia m)
        {
            m.Angle = NormalizeAngle(m.Angle);

            return m;
        }

        public static float SqrLength(Minutia m1, Minutia m2)
        {
            return (float)(Math.Pow(m1.X - m2.X, 2) + Math.Pow(m1.Y - m2.Y, 2));
        }
    }
}
