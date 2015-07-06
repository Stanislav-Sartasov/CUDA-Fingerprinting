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
            var image = Properties.Resources.SampleFinger;
            Matrix M = new Matrix();
            Bitmap newPic = M.calculatingAverageColor(image);
            newPic.Save("newPic.bmp");
        }
    }
}
