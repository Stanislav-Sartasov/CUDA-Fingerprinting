using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common.Matrix;
using System.Drawing;

namespace CUDAFingerprinting.Common.Matrix.Tests
{
    [TestClass]
    public class MatrixTest
    {
        [TestMethod]
        public void SimpleMatrixTest()
        {
            String filename = "1_1.bmp";
            Matrix M = new Matrix();
            M.calculatingAverageColor(filename);
        }
    }
}
