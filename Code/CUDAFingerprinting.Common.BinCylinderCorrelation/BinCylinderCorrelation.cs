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
            uint[] cylinder1Validities, uint[] cylinder2Validities,
            uint minMatchableElementsCount)
        {
            uint[] commonValidities = cylinder1Validities.Zip(cylinder2Validities, (first, second) => first & second).ToArray();

            uint[] c1GivenCommon = linearizedCylinder1.Zip(commonValidities, (first, second) => first & second).ToArray();
            uint[] c2GivenCommon = linearizedCylinder2.Zip(commonValidities, (first, second) => first & second).ToArray();

            double c1GivenCommonNorm = Math.Sqrt(GetOneBitsCount(c1GivenCommon));
            double c2GivenCommonNorm = Math.Sqrt(GetOneBitsCount(c2GivenCommon));

            bool matchable = true;

            var matchableElementsCount = GetOneBitsCount(commonValidities);

            // To be done later (cylinder matching conditions, min interminutiae angle not implemented)
            if (/* matchableElementsCount >= minMatchableElementsCount || */
                c1GivenCommonNorm + c2GivenCommonNorm == 0)
            {
                matchable = false;
            }

            double correlation = 0;
            if (matchable)
            {
                uint givenXOR1 = c1GivenCommon[0] ^ c2GivenCommon[0];
                uint[] givenXOR = c1GivenCommon.Zip(c2GivenCommon, (first, second) => first ^ second).ToArray();
                double givenXORNorm = Math.Sqrt(GetOneBitsCount(givenXOR));
                correlation = 1 - givenXORNorm / (c1GivenCommonNorm + c2GivenCommonNorm);
            }

            return correlation;
        }
    }
}
