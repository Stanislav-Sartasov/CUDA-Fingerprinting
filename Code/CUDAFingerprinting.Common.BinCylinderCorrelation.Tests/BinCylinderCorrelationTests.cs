using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.Common.BinCylinderCorrelation.Tests
{
    [TestClass]
    public class BinCylinderCorrelationTests
    {
        public static int[] GetValidity(ref int[] linearizedCylinder)
        {
            int[] cylinderValidity = new int[linearizedCylinder.Length];
            for (int i = 0; i < linearizedCylinder.Length; i++)
            {
                if (linearizedCylinder[i] == -1)
                {
                    linearizedCylinder[i] = 0;
                }
                else
                {
                    cylinderValidity[i] = 1;
                }
            }

            return cylinderValidity;
        }

        public static int[] Linearize(int[, ,] cylinder)
        {
            int cylinderY = cylinder.GetLength(0);
            int cylinderX = cylinder.GetLength(1);
            int[] linearizedCylinder = new int[cylinderY * cylinderX * cylinderX];

            for (int i = 0; i < cylinderY; i++)
            {
                for (int j = 0; j < cylinderX; j++)
                {
                    for (int k = 0; k < cylinderX; k++)
                    {
                        linearizedCylinder[i * cylinderX * cylinderX + j * cylinderX + k] = cylinder[i, j, k];
                    }
                }
            }

            return linearizedCylinder;
        }

        [TestMethod]
        public void TestBinCylinderCorrelation()
        {
            // Given
            int[, ,] cylinderZeros = new int[,,]
            {
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 }, 
                    { 0, 0, 0 }
                },
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 }, 
                    { 0, 0, 0 }
                }
            };

            int[, ,] cylinderOnes = new int[,,]
            {
                {
                    { 1, 1, 1 },
                    { 1, 1, 1 }, 
                    { 1, 1, 1 }
                },
                {
                    { 1, 1, 1 },
                    { 1, 1, 1 }, 
                    { 1, 1, 1 }
                }
            };

            int[, ,] cylinderMixed = new int[,,]
            {
                
                {
                    { 1, 1, -1 },
                    { 1, 0, 0 }, 
                    { -1, 1, 0 }
                },
                {
                    { 1, 0, 1 },
                    { -1, -1, -1 }, 
                    { 0, 1, 1 }
                }
            };

            int[][] linearizedCylinders = 
            { 
                Linearize(cylinderZeros), Linearize(cylinderOnes), Linearize(cylinderMixed)
            };

            int[][] cylinderValidities = 
            {
                GetValidity(ref linearizedCylinders[0]), 
                GetValidity(ref linearizedCylinders[1]), 
                GetValidity(ref linearizedCylinders[2])   
            };

            // When
            var correlation0 = BinCylinderCorrelation.GetBinCylinderCorrelation(
                linearizedCylinders[0], linearizedCylinders[1],
                cylinderValidities[0], cylinderValidities[1], 0); // Min matching elements count not implemented/tested thus far

            var correlation1 = BinCylinderCorrelation.GetBinCylinderCorrelation(
                linearizedCylinders[1], linearizedCylinders[2],
                cylinderValidities[1], cylinderValidities[2], 0);

            var correlation2 = BinCylinderCorrelation.GetBinCylinderCorrelation(
                linearizedCylinders[2], linearizedCylinders[2],
                cylinderValidities[2], cylinderValidities[2], 0);

            var correlation3 = BinCylinderCorrelation.GetBinCylinderCorrelation(
                linearizedCylinders[1], linearizedCylinders[1],
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
