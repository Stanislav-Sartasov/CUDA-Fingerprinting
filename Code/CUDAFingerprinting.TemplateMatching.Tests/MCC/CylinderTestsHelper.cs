using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.TemplateMatching.MCC.Tests
{
    class CylinderTestsHelper
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


        static public int[, ,] cylinderZerosValues = new int[,,]
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

        static public int[, ,] cylinderOnesValues = new int[,,]
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

        static public int[, ,] cylinderMixedValues = new int[,,]
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
        static public uint[][] linearizedCylinders = 
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
        static public Cylinder cylinderZeros =
            new Cylinder(linearizedCylinders[0], Math.PI / 6, Math.Sqrt(CylinderHelper.GetOneBitsCount(linearizedCylinders[0])));
        static public Cylinder cylinderOnes =
            new Cylinder(linearizedCylinders[1], Math.PI / 4, Math.Sqrt(CylinderHelper.GetOneBitsCount(linearizedCylinders[1])));
        static public Cylinder cylinderMixed =
            new Cylinder(linearizedCylinders[2], Math.PI / 3, Math.Sqrt(CylinderHelper.GetOneBitsCount(linearizedCylinders[2])));

        // Stupid version
        //Cylinder cylinderZeros =
        //    new Cylinder(linearizedCylinders[0], Math.PI / 6, BinCylinderCorrelation.CalculateCylinderNorm(linearizedCylinders[0]));
        //Cylinder cylinderOnes =
        //    new Cylinder(linearizedCylinders[1], Math.PI / 4, BinCylinderCorrelation.CalculateCylinderNorm(linearizedCylinders[1]));
        //Cylinder cylinderMixed =
        //    new Cylinder(linearizedCylinders[2], Math.PI / 3, BinCylinderCorrelation.CalculateCylinderNorm(linearizedCylinders[2]));

    }
}