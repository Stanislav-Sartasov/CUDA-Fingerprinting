using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.TemplateMatching.MCC.Tests
{
    [TestClass]
    public class BinCylinderSimilarityTests
    {
        public static uint[] GetValidities(int[, ,] cylinder)
        {
            int cylinderY = cylinder.GetLength(0);
            int cylinderX = cylinder.GetLength(1);
            uint[] cylinderValidity = new uint[cylinderY * cylinderX * cylinderX];

            for (int i = 0; i < cylinderY; i++)
            {
                for (int j = 0; j < cylinderX; j++)
                {
                    for (int k = 0; k < cylinderX; k++)
                    {
                        if (cylinder[i, j, k] == 0 || cylinder[i, j, k] == 1)
                        {
                            cylinderValidity[i * cylinderX * cylinderX + j * cylinderX + k] = 1;
                        }
                        else if (cylinder[i, j, k] != -1)
                        {
                            throw new Exception("Invalid input cylinder");
                        }
                    }
                }
            }

            return cylinderValidity;
        }

        [TestMethod]
        public void TestBinCylinderSimilarity()
        {
            uint[][] cylinderValidities = 
        {
            CylinderTestsHelper.ConvertArrayUintToBinary(GetValidities(CylinderTestsHelper.cylinderZerosValues)), 
            CylinderTestsHelper.ConvertArrayUintToBinary(GetValidities(CylinderTestsHelper.cylinderOnesValues)), 
            CylinderTestsHelper.ConvertArrayUintToBinary(GetValidities(CylinderTestsHelper.cylinderMixedValues))
        };

            // When
            var correlation0 = BinCylinderSimilarity.GetCylinderSimilarity(
                CylinderTestsHelper.linearizedCylinders[0], CylinderTestsHelper.linearizedCylinders[1],
                cylinderValidities[0], cylinderValidities[1], 0);

            var correlation1 = BinCylinderSimilarity.GetCylinderSimilarity(
                CylinderTestsHelper.linearizedCylinders[1], CylinderTestsHelper.linearizedCylinders[2],
                cylinderValidities[1], cylinderValidities[2], 0);

            var correlation2 = BinCylinderSimilarity.GetCylinderSimilarity(
                CylinderTestsHelper.linearizedCylinders[2], CylinderTestsHelper.linearizedCylinders[2],
                cylinderValidities[2], cylinderValidities[2], 0);

            var correlation3 = BinCylinderSimilarity.GetCylinderSimilarity(
                CylinderTestsHelper.linearizedCylinders[1], CylinderTestsHelper.linearizedCylinders[1],
                cylinderValidities[1], cylinderValidities[1], 0);

            // Then
            Assert.AreEqual(correlation0, 0.0);
            Assert.IsTrue(correlation1 > 0.5 && correlation1 < 1.0); // Around 0.65 is a correct value
            Assert.AreEqual(correlation2, 1.0);
            Assert.AreEqual(correlation3, 1.0);

            Console.WriteLine(correlation0 + "; " + correlation1 + "; " + correlation2 + "; " + correlation3);
        }
    }
}
