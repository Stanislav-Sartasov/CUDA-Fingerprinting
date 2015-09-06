using System;

namespace CUDAFingerprinting.TemplateMatching.MCC
{
    public class Cylinder
    {
        public uint[] Values { get; set; }
        public double Angle { get; set; }
        public double Norm { get; set; }

        public Cylinder() { }

        public Cylinder(uint[] givenValues, double givenAngle, double givenNorm)
        {
            Values = givenValues;
            Angle = givenAngle;
            Norm = givenNorm;
        }
    }

    public class Template
    {
        public Cylinder[] Cylinders { get; private set; }
        public Template(Cylinder[] givenCylinders)
        {
            Cylinders = givenCylinders;
        }
    }

    public class CylinderDatabase
    {
        public Cylinder[] Cylinders { get; private set; }
        public uint[] TemplateIndices { get; private set; }

        public CylinderDatabase(Cylinder[] givenCylinders, uint[] givenTemplateIndices)
        {
            Cylinders = givenCylinders;
            TemplateIndices = givenTemplateIndices;
        }
    }

    public static class CylinderHelper
    {
        public static double CalculateCylinderNorm(uint[] cylinder)
        {
            //not for bit-based implementation
            int sum = 0;
            for (int i = 0; i < cylinder.Length; i++)
            {
                if (cylinder[i] == 1)
                {
                    sum++;
                }
            }
            return Math.Sqrt(sum);
        }

        public static uint GetOneBitsCount(uint[] arr)
        {
            uint[] _arr = (uint[])arr.Clone();
            uint count = 0;
            for (int i = 0; i < _arr.Length; i++)
            {
                for (int j = 31; j >= 0; j--)
                {
                    if (_arr[i] % 2 == 1)
                    {
                        count++;
                    }
                    _arr[i] /= 2;
                }
            }
            return count;
        }

        public static double GetAngleDiff(double angle1, double angle2)
        {
            double diff = angle1 - angle2;
            double res =
                diff < -Math.PI ? diff + 2 * Math.PI :
                diff >= Math.PI ? diff - 2 * Math.PI :
                diff;
            return res;
        }
    }
}
