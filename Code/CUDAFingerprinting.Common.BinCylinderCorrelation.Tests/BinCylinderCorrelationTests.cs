using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.Common.BinCylinderCorrelation.Tests
{
    [TestClass]
    public class BinCylinderCorrelationTests
    {
        public static uint[] ConvertArrayUintToBinary(uint[] intArray)
        {
            uint[] binaryArray = new uint[(intArray.Length + 32 + 1) / 32]; // Same as ceilMod macro in GPU Solution

            for (int i = 0; i < intArray.Length; i++)
            {
                if (intArray[i] == 1)
                {
                    binaryArray[i / 32] += (uint)Math.Pow(2, (32 - 1 - i % 32));
                }
                else if (intArray[i] != 0)
                {
                    throw new Exception("Invalid uintToBinary convertion input");
                }
            }

            return binaryArray;
        }

        public static uint[] ConvertArrayBinaryToUint(uint[] binaryArray)
        {
            uint[] intArray = new uint[binaryArray.Length * 32]; // Same as ceilMod macro in GPU Solution

            for (int i = 0; i < binaryArray.Length; i++)
            {
                for (int j = 31; j >= 0; j--)
                {
                    if (binaryArray[i] % 2 == 1)
                    {
                        intArray[i * 32 + j] = 1;
                    }
                    binaryArray[i] /= 2;
                }
            }

            return intArray;
        }

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

        public static uint[] Linearize(int[, ,] cylinder)
        {
            int cylinderY = cylinder.GetLength(0);
            int cylinderX = cylinder.GetLength(1);
            uint[] linearizedCylinder = new uint[cylinderY * cylinderX * cylinderX];

            for (int i = 0; i < cylinderY; i++)
            {
                for (int j = 0; j < cylinderX; j++)
                {
                    for (int k = 0; k < cylinderX; k++)
                    {
                        if (cylinder[i, j, k] == -1 || cylinder[i, j, k] == 0)
                        {
                            linearizedCylinder[i * cylinderX * cylinderX + j * cylinderX + k] = 0;
                        }
                        else if (cylinder[i, j, k] == 1)
                        {
                            linearizedCylinder[i * cylinderX * cylinderX + j * cylinderX + k] = 1;
                        }
                        else
                        {
                            throw new Exception("Invalid input cylinder");
                        }
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

            uint[][] linearizedCylinders = 
            { 
                ConvertArrayUintToBinary(Linearize(cylinderZeros)), 
                ConvertArrayUintToBinary(Linearize(cylinderOnes)), 
                ConvertArrayUintToBinary(Linearize(cylinderMixed))
            };

            uint[][] cylinderValidities = 
            {
                ConvertArrayUintToBinary(GetValidities(cylinderZeros)), 
                ConvertArrayUintToBinary(GetValidities(cylinderOnes)), 
                ConvertArrayUintToBinary(GetValidities(cylinderMixed))
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