using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common.BinCylinderCorrelation
{
    public class BinCylinderCorrelation
    {
        public static double GetBinCylinderCorrelation(
            int[] linearizedCylinder1, int[] linearizedCylinder2,
            int[] cylinder1Validities, int[] cylinder2Validities,
            int minMatchableElementsCount)
        {
            var commonValidities = cylinder1Validities.Zip(cylinder2Validities, (first, second) => first & second).ToArray();

            int[] c1GivenCommon = linearizedCylinder1.Zip(commonValidities, (first, second) => first & second).ToArray();
            int[] c2GivenCommon = linearizedCylinder2.Zip(commonValidities, (first, second) => first & second).ToArray();

            var c1GivenCommonNorm = Math.Sqrt(c1GivenCommon.Sum()); // Is this cast necessary?
            var c2GivenCommonNorm = Math.Sqrt(c2GivenCommon.Sum());

            bool matchable = true;

            var matchableElementsCount = commonValidities.Sum();

            // To be done later (cylinder matching conditions, min interminutiae angle not implemented)
            //if (matchableElementsCount >= minMatchableElementsCount || 
            //    c1GivenCommonNorm + c2GivenCommonNorm == 0)
            //{
            //    matchable = false;
            //}

            double correlation = 0;
            if (matchable)
            {
                int[] givenXOR = c1GivenCommon.Zip(c2GivenCommon, (first, second) => first ^ second).ToArray();
                var givenXORNorm = Math.Sqrt(givenXOR.Sum());
                correlation = 1 - givenXORNorm / (c1GivenCommonNorm + c2GivenCommonNorm);
            }

            return correlation;
        }
    }
}
