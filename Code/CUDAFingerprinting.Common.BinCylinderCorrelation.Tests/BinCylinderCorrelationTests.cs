using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.Common.BinCylinderCorrelation.Tests
{
    //public class Cylinder3d
    //{
    //    public uint[, ,] Values { get; private set; }
    //    public double Angle { get; private set; }
    //    public double Norm { get; private set; }

    //    public Cylinder3d(uint[, ,] givenValues, double givenAngle, double givenNorm)
    //    {
    //        Values = givenValues;
    //        Angle = givenAngle;
    //        Norm = givenNorm;
    //    }
    //}

    //public class Cylinder3dDb : Cylinder3d
    //{
    //    public uint TemplateIndex { get; private set; }

    //    public Cylinder3dDb(uint[, ,] givenValues, double givenAngle, double givenNorm, double givenTemplateIndex) 
    //        : base (givenValues, givenAngle, givenNorm);
    //    {
    //         TemplateIndex = givenTemplateIndex;
    //    }
    //}

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
            int[, ,] cylinderZerosValues = new int[,,]
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

            int[, ,] cylinderOnesValues = new int[,,]
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

            int[, ,] cylinderMixedValues = new int[,,]
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

            // Bitwise version
            //uint[][] linearizedCylinders = 
            //{ 
            //    ConvertArrayUintToBinary(Linearize(cylinderZerosValues)), 
            //    ConvertArrayUintToBinary(Linearize(cylinderOnesValues)), 
            //    ConvertArrayUintToBinary(Linearize(cylinderMixedValues))
            //};

            // Stupid version
            uint[][] linearizedCylinders = 
            { 
                Linearize(cylinderZerosValues), 
                Linearize(cylinderOnesValues), 
                Linearize(cylinderMixedValues)
            };

            // Hardcoding angles
            Cylinder cylinderZeros = 
                new Cylinder(linearizedCylinders[0], 0.83, BinCylinderCorrelation.CalculateCylinderNorm(linearizedCylinders[0]));
            Cylinder cylinderOnes = 
                new Cylinder(linearizedCylinders[1], 1.54, BinCylinderCorrelation.CalculateCylinderNorm(linearizedCylinders[1]));
            Cylinder cylinderMixed = 
                new Cylinder(linearizedCylinders[2], 1.77, BinCylinderCorrelation.CalculateCylinderNorm(linearizedCylinders[2]));



            Template query = new Template(new Cylinder[]
            {
                cylinderZeros, cylinderOnes
            });

            //TemplateDb[] db = new TemplateDb[]
            //{
            //    new TemplateDb(new CylinderDb[]
            //    {
                    
            //    })
            //};

            // When
            //var correlation0 = BinCylinderCorrelation.GetTemplateCorrelation(
            //    linearizedCylinders[0], linearizedCylinders[1]); // Min matching elements count not implemented/tested thus far

            //var correlation1 = BinCylinderCorrelation.GetTemplateCorrelation(
            //    linearizedCylinders[1], linearizedCylinders[2]);

            //var correlation2 = BinCylinderCorrelation.GetTemplateCorrelation(
            //    linearizedCylinders[2], linearizedCylinders[2]);

            //var correlation3 = BinCylinderCorrelation.GetTemplateCorrelation(
            //    linearizedCylinders[1], linearizedCylinders[1]);

            // Then
            //Assert.AreEqual(correlation0, 0.0);
            //Assert.IsTrue(correlation1 > 0.5 && correlation1 < 1.0); // Around 0.65 is a correct value
            //Assert.AreEqual(correlation2, 1.0);
            //Assert.AreEqual(correlation3, 1.0);

            //Console.WriteLine(correlation0 + "; " + correlation1 + "; " + correlation2 + "; " + correlation3);
        }
    }
}