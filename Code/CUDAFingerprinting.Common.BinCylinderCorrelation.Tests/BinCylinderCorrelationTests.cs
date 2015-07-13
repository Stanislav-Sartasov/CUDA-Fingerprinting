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
            uint[][] linearizedCylinders = 
            { 
                ConvertArrayUintToBinary(Linearize(cylinderZerosValues)), 
                ConvertArrayUintToBinary(Linearize(cylinderOnesValues)), 
                ConvertArrayUintToBinary(Linearize(cylinderMixedValues))
            };

            // Stupid version
            //uint[][] linearizedCylinders = 
            //{ 
            //    Linearize(cylinderZerosValues), 
            //    Linearize(cylinderOnesValues), 
            //    Linearize(cylinderMixedValues)
            //};


            // Bitwise version 
            Cylinder cylinderZeros =
                new Cylinder(linearizedCylinders[0], Math.PI / 6, Math.Sqrt(BinCylinderCorrelation.GetOneBitsCount(linearizedCylinders[0])));
            Cylinder cylinderOnes =
                new Cylinder(linearizedCylinders[1], Math.PI / 4, Math.Sqrt(BinCylinderCorrelation.GetOneBitsCount(linearizedCylinders[1])));
            Cylinder cylinderMixed =
                new Cylinder(linearizedCylinders[2], Math.PI / 3, Math.Sqrt(BinCylinderCorrelation.GetOneBitsCount(linearizedCylinders[2])));

            // Stupid version
            //Cylinder cylinderZeros =
            //    new Cylinder(linearizedCylinders[0], Math.PI / 6, BinCylinderCorrelation.CalculateCylinderNorm(linearizedCylinders[0]));
            //Cylinder cylinderOnes =
            //    new Cylinder(linearizedCylinders[1], Math.PI / 4, BinCylinderCorrelation.CalculateCylinderNorm(linearizedCylinders[1]));
            //Cylinder cylinderMixed =
            //    new Cylinder(linearizedCylinders[2], Math.PI / 3, BinCylinderCorrelation.CalculateCylinderNorm(linearizedCylinders[2]));


            Template query = new Template(new Cylinder[]
            {
                cylinderMixed
            });

            Template[] db = new Template[]
            {
                new Template(new Cylinder[]
                {
                    cylinderOnes
                }),

                new Template(new Cylinder[]
                {
                    cylinderOnes,
                    cylinderMixed
                }),

                new Template(new Cylinder[]
                {
                    cylinderMixed,
                    cylinderMixed,
                    cylinderMixed,
                    cylinderMixed
                })
            };

            Cylinder[] contiguousArr = new Cylinder[]
            {
                cylinderOnes, cylinderOnes, cylinderMixed, cylinderMixed, cylinderMixed, cylinderMixed, cylinderMixed
            };
            uint[] templateIndices = new uint[] { 0, 1, 1, 2, 2, 2, 2 };
            CylinderDatabase cylinderDb = new CylinderDatabase(contiguousArr, templateIndices);
            int[] dbTemplatesLengths = new int[] { 1, 2, 4 };


            double[] similarityRates = BinCylinderCorrelation.GetTemplateCorrelation(query, db); // templateDb version
            //double[] similarityRates = BinCylinderCorrelation.GetTemplateCorrelationOptimized(query, cylinderDb, dbTemplatesLengths); // cylinderDb version
            for (int i = 0; i < similarityRates.Length; i++)
            {
                Console.Write(similarityRates[i] + " ");
            }
            Console.WriteLine();
        }
    }
}