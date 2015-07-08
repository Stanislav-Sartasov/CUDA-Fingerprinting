using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices;

namespace CUDAFingerprinting.GPU.BinCylinderCorrelation.Tests
{
    [TestClass]
    public class BinCylinderCorrelationTests
    {
        [DllImport("CUDAFingerprinting.GPU.BinCylinderCorrelation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "getBinCylinderCorrelation")]
        public static extern float getBinCylinderCorrelation(
            uint cylinderCapacity,
            uint[] cudaCylinder1, uint[] cudaCylinder2,
            uint[] cudaValidities1, uint[] cudaValidities2);

        [TestMethod]
        public void TestCorrelationZeros()
        {
            uint[] linearizedCylinder1 = { 0 };
            uint[] cylinder1Validities = { Convert.ToUInt32("11111111111111111100000000000000", 2) };

            uint[] linearizedCylinder2 = { 0 };
            uint[] cylinder2Validities = { Convert.ToUInt32("11111111111111111100000000000000", 2) };

            float correlation =
                getBinCylinderCorrelation(1, linearizedCylinder1, linearizedCylinder2, cylinder1Validities, cylinder2Validities);

            Console.WriteLine("Correlation: " + correlation);

            // They are not even matchable
            Assert.AreEqual(correlation, 0.0);

        }

        [TestMethod]
        public void TestCorrelationOnes()
        {
            uint[] linearizedCylinder1 = { Convert.ToUInt32("11111111111111111100000000000000", 2) };
            uint[] cylinder1Validities = { Convert.ToUInt32("11111111111111111100000000000000", 2) };

            uint[] linearizedCylinder2 = { Convert.ToUInt32("11111111111111111100000000000000", 2) };
            uint[] cylinder2Validities = { Convert.ToUInt32("11111111111111111100000000000000", 2) };

            float correlation =
                getBinCylinderCorrelation(1, linearizedCylinder1, linearizedCylinder2, cylinder1Validities, cylinder2Validities);

            Console.WriteLine("Correlation: " + correlation);

            Assert.AreEqual(correlation, 1.0);

        }

        [TestMethod]
        public void TestCorrelationRandom()
        {
            uint[] linearizedCylinder1 = { Convert.ToUInt32("11111111111111111100000000000000", 2) };
            uint[] cylinder1Validities = { Convert.ToUInt32("11111111111111111100000000000000", 2) };

            uint[] linearizedCylinder2 = { Convert.ToUInt32("11010001010100001100000000000000", 2) };
            uint[] cylinder2Validities = { Convert.ToUInt32("11011101111100011100000000000000", 2) };

            float correlation =
                getBinCylinderCorrelation(1, linearizedCylinder1, linearizedCylinder2, cylinder1Validities, cylinder2Validities);

            Console.WriteLine("Correlation: " + correlation);

            Assert.IsTrue(correlation > 0.65 && correlation < 0.66);
        }
    }
}