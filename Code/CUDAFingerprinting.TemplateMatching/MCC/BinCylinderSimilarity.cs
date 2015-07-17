using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.TemplateMatching.MCC
{
    public class BinCylinderSimilarity
    {
        public static double GetCylinderSimilarity(
            uint[] linearizedCylinder1, uint[] linearizedCylinder2,
            uint[] cylinder1Validities, uint[] cylinder2Validities,
            uint minMatchableElementsCount)
        {
            uint[] commonValidities = cylinder1Validities.Zip(cylinder2Validities, (first, second) => first & second).ToArray();

            uint[] c1GivenCommon = linearizedCylinder1.Zip(commonValidities, (first, second) => first & second).ToArray();
            uint[] c2GivenCommon = linearizedCylinder2.Zip(commonValidities, (first, second) => first & second).ToArray();

            double c1GivenCommonNorm = Math.Sqrt(CylinderHelper.GetOneBitsCount(c1GivenCommon));
            double c2GivenCommonNorm = Math.Sqrt(CylinderHelper.GetOneBitsCount(c2GivenCommon));

            bool matchable = true;

            var matchableElementsCount = CylinderHelper.GetOneBitsCount(commonValidities);

            // To be done later (cylinder matching conditions, min interminutiae angle not implemented)
            if (/* matchableElementsCount >= minMatchableElementsCount || */
                c1GivenCommonNorm + c2GivenCommonNorm == 0)
            {
                matchable = false;
            }

            double correlation = 0;
            if (matchable)
            {
                uint[] givenXOR = c1GivenCommon.Zip(c2GivenCommon, (first, second) => first ^ second).ToArray();
                double givenXORNorm = Math.Sqrt(CylinderHelper.GetOneBitsCount(givenXOR));
                correlation = 1 - givenXORNorm / (c1GivenCommonNorm + c2GivenCommonNorm);
            }

            return correlation;
        }
    }
}