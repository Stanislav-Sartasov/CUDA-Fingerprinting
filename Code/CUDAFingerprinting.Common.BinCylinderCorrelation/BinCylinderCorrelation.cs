using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common.BinCylinderCorrelation
{
    public class BinCylinderCorrelation
    {
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

        public static double GetBinCylinderCorrelation(
            uint[] linearizedCylinder1, uint[] linearizedCylinder2,
            uint minMatchableElementsCount)
        {
            double c1Norm = Math.Sqrt(GetOneBitsCount(linearizedCylinder1));
            double c2Norm = Math.Sqrt(GetOneBitsCount(linearizedCylinder2));

            double correlation = 0;
            if (c1Norm + c2Norm != 0)
            {
                uint[] givenXOR = linearizedCylinder1.Zip(linearizedCylinder2, (first, second) => first ^ second).ToArray();
                double givenXORNorm = Math.Sqrt(GetOneBitsCount(givenXOR));
                correlation = 1 - givenXORNorm / (c1Norm + c2Norm);
            }

            return correlation;
        }
    }
}