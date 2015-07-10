using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common.Matrix;
using System.Drawing;
using System.IO;

namespace CUDAFingerprinting.Common.Matrix.Tests
{
    [TestClass]
    public class MatrixTest
    {
        [TestMethod]
        public void SimpleMatrixTest()
        {
            var image = Properties.Resources.SampleFinger;
            Matrix M = new Matrix(image);

            double [,] matrix = M.SobelFilter();
            M.MatrixMaking();

            Bitmap newPic = M.BWPicture();
            newPic.Save("newPic.bmp");
        }
    }
}